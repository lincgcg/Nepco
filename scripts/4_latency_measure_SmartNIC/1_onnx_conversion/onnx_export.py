import os
import argparse
import torch
import torch.nn as nn
import sys
from pathlib import Path

AE_ROOT = Path(__file__).resolve().parents[3]
MODEL_FINETUNE_DIR = AE_ROOT / "scripts" / "2_model_train" / "model_finetune"
sys.path.insert(0, str(AE_ROOT))
sys.path.insert(0, str(MODEL_FINETUNE_DIR))

from uer.utils.config import load_hyperparam
from uer.opts import finetune_opts, tokenizer_opts
from uer.utils import str2tokenizer
from uer.utils.misc import pooling

from finetune import Classifier, read_dataset

try:
    import onnx
    import onnxruntime
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
    from onnxruntime.quantization.preprocess import quant_pre_process
except ImportError:
    raise ImportError("Please install onnx and onnxruntime: pip install onnx onnxruntime")


class OnnxExportWrapper(nn.Module):
    """
    Keep only the inference inputs (src, seg) and remove tgt from the graph.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src, seg):
        emb = self.model.embedding(src, seg)
        output = self.model.encoder(emb, seg)
        output = pooling(output, seg, self.model.pooling_type)
        output = torch.tanh(self.model.output_layer_1(output))
        logits = self.model.output_layer_2(output)
        return logits


class RealDataReader(CalibrationDataReader):
    """
    Use real training samples for static INT8 calibration.
    """
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.offset = 0
        self.max_samples = 200

    def get_next(self):
        if self.offset >= len(self.dataset) or self.offset >= self.max_samples:
            return None

        sample = self.dataset[self.offset]
        self.offset += 1

        src = torch.LongTensor([sample[0]])
        seg = torch.LongTensor([sample[2]])

        return {
            "src": src.numpy(),
            "seg": seg.numpy()
        }


def export_onnx(args):
    if not os.path.exists(args.output_model_path):
        print(f"Error: trained model file not found: {args.output_model_path}")
        return

    os.makedirs(args.onnx_path, exist_ok=True)

    print("[-] Building model...")
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    model = Classifier(args)

    print(f"[-] Loading checkpoint from {args.output_model_path}")
    model.load_state_dict(torch.load(args.output_model_path, map_location="cpu"), strict=False)

    real_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    real_model.eval()
    real_model.cpu()

    if hasattr(real_model, "encoder") and hasattr(real_model.encoder, "prepare_for_onnx_export"):
        real_model.encoder.prepare_for_onnx_export()

    wrapped_model = OnnxExportWrapper(real_model)
    wrapped_model.eval()
    wrapped_model.cpu()

    print(f"[-] Creating dummy input with seq_length={args.seq_length}...")
    dummy_src = torch.randint(0, len(args.tokenizer.vocab), (1, args.seq_length), dtype=torch.long)
    dummy_seg = torch.ones((1, args.seq_length), dtype=torch.long)

    onnx_path = os.path.join(args.onnx_path, "output.onnx")

    print(f"[-] Exporting to standard ONNX: {onnx_path}")
    with torch.inference_mode():
        torch.onnx.export(
            wrapped_model,
            (dummy_src, dummy_seg),
            onnx_path,
            input_names=["src", "seg"],
            output_names=["logits"],
            dynamic_axes={
                "src": {0: "batch_size"},
                "seg": {0: "batch_size"},
                "logits": {0: "batch_size"}
            },
            opset_version=17,
            do_constant_folding=True
        )

    print("[-] Checking exported ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    print("[-] Pre-processing for BF3 optimization (fusing operators)...")
    preprocessed_path = os.path.join(args.onnx_path, "output.preprocessed.onnx")
    quant_pre_process(input_model_path=onnx_path, output_model_path=preprocessed_path)

    print("[-] Performing Static Quantization (Int8)...")
    print(f"    Reading calibration data from: {args.train_path}")

    train_ds = read_dataset(args, args.train_path)
    dr = RealDataReader(train_ds)

    quantized_path = os.path.join(args.onnx_path, "output.static_int8.onnx")

    quantize_static(
        model_input=preprocessed_path,
        model_output=quantized_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )

    print("=" * 50)
    print("[SUCCESS] Model export finished.")
    print(f"1. Standard ONNX model: {onnx_path}")
    print(f"2. BF3 INT8 model: {quantized_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Reuse model/training options, but ONNX export does not need a dev set.
    finetune_opts(parser, require_train_dev=False)
    tokenizer_opts(parser)

    parser.add_argument("--soft_targets", action="store_true")
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    parser.add_argument("--labels_num", type=int, default=2)
    parser.add_argument("--onnx_path", type=str)

    args = parser.parse_args()
    args = load_hyperparam(args)

    if not args.train_path:
        raise ValueError("--train_path is required for ONNX calibration.")

    export_onnx(args)

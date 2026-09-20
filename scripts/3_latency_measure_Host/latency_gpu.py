import sys
import os
import time
import math
import argparse
import statistics
import torch
import torch.nn as nn

AE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, AE_ROOT)

from uer.embeddings import *
from uer.encoders import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.utils.misc import pooling
from uer.opts import finetune_opts, tokenizer_opts


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)

        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling_type = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha

        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def infer(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = pooling(output, seg, self.pooling_type)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        return logits

    def forward(self, src, tgt, seg, soft_tgt=None):
        logits = self.infer(src, seg)

        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict

    has_module_prefix = all(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict

    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[len("module."):]] = v
    return new_state_dict


def load_finetuned_checkpoint(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = strip_module_prefix(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print("[Warning] Missing keys:")
        for k in missing:
            print("   ", k)

    if unexpected:
        print("[Warning] Unexpected keys:")
        for k in unexpected:
            print("   ", k)


def prepare_model_for_fast_test(model):
    """
    Try to enable the fast inference path for the current CNN encoder.
    """
    real_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    real_model.eval()

    if hasattr(real_model, "encoder"):
        encoder = real_model.encoder
        if hasattr(encoder, "prepare_for_onnx_export"):
            encoder.prepare_for_onnx_export()
        elif hasattr(encoder, "refresh_infer_cache"):
            encoder.refresh_infer_cache()
        elif hasattr(encoder, "update_softmax_weights"):
            encoder.update_softmax_weights()

    return real_model


def read_dataset(args, path):
    dataset, columns = [], {}
    pad_id = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]

    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue

            line = line.rstrip("\r\n").split("\t")
            tgt = int(line[columns["label"]])

            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]

            if "text_b" not in columns:
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN]
                )
                seg = [1] * len(src)
            else:
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN]
                )
                src_b = args.tokenizer.convert_tokens_to_ids(
                    args.tokenizer.tokenize(text_b) + [SEP_TOKEN]
                )
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]

            if len(src) < args.seq_length:
                pad_len = args.seq_length - len(src)
                src.extend([pad_id] * pad_len)
                seg.extend([0] * pad_len)

            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))

    return dataset


def build_test_tensors(args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    if args.device.type == "cuda":
        src = src.pin_memory()
        tgt = tgt.pin_memory()
        seg = seg.pin_memory()

    return src, tgt, seg


def batch_loader(batch_size, src, tgt, seg):
    instances_num = src.size(0)
    for i in range(0, instances_num, batch_size):
        yield (
            src[i: i + batch_size],
            tgt[i: i + batch_size],
            seg[i: i + batch_size],
        )


def evaluate_once(args, dataset, compute_metrics=False, istest=False):
    """
    Run one full inference pass.
    The measured time only covers args.model.infer(src_batch, seg_batch).
    If compute_metrics is true, accuracy and confusion matrix are also computed.
    """
    src, tgt, seg = build_test_tensors(args, dataset)

    batch_size = args.batch_size
    device = args.device

    total_time = 0.0
    correct = 0

    if compute_metrics:
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long, device=device)
    else:
        confusion = None

    args.model.eval()
    prepare_model_for_fast_test(args.model)

    with torch.inference_mode():
        for src_batch, tgt_batch, seg_batch in batch_loader(batch_size, src, tgt, seg):
            src_batch = src_batch.to(device, non_blocking=(device.type == "cuda"))
            tgt_batch = tgt_batch.to(device, non_blocking=(device.type == "cuda"))
            seg_batch = seg_batch.to(device, non_blocking=(device.type == "cuda"))

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_t = time.perf_counter()

            logits = args.model.infer(src_batch, seg_batch)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end_t = time.perf_counter()

            total_time += (end_t - start_t)

            if compute_metrics:
                pred = torch.argmax(logits, dim=1)
                gold = tgt_batch

                correct += torch.sum(pred == gold).item()

                indices = pred * args.labels_num + gold
                bincount = torch.bincount(indices, minlength=args.labels_num * args.labels_num)
                confusion += bincount.view(args.labels_num, args.labels_num)

    per_sample_time = total_time / len(dataset)

    if not compute_metrics:
        return {
            "total_time": total_time,
            "per_sample_time": per_sample_time,
            "acc": None,
            "confusion": None,
        }

    confusion_cpu = confusion.cpu()
    acc = correct / len(dataset)

    if istest:
        print(confusion_cpu)
        eps = 1e-9
        prf_dir = os.path.join(os.path.dirname(args.output_model_path), "prf")
        os.makedirs(prf_dir, exist_ok=True)
        filename = os.path.join(prf_dir, "prf.csv")

        with open(filename, "w+", encoding="utf-8") as f2:
            f2.write("Label num,Precision,Recall,F1\n")
            for i in range(confusion_cpu.size(0)):
                p = confusion_cpu[i, i].item() / (confusion_cpu[i, :].sum().item() + eps)
                r = confusion_cpu[i, i].item() / (confusion_cpu[:, i].sum().item() + eps)
                f1 = 2 * p * r / (p + r + eps)

                args.logger.info("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
                f2.write("{},{},{},{}\n".format(i, p, r, f1))

    args.logger.info("Acc. (Correct/Total): {:.4f} ({}/{})".format(acc, correct, len(dataset)))

    return {
        "total_time": total_time,
        "per_sample_time": per_sample_time,
        "acc": acc,
        "confusion": confusion_cpu,
    }


def evaluate_with_ci(args, dataset, istest=False, repeat_times=5):
    """
    Run one untimed warmup pass, then repeat timed inference passes.
    Report the mean per-sample latency with a 95% confidence interval.
    """
    print("Warmup...")
    _ = evaluate_once(args, dataset, compute_metrics=False, istest=False)

    per_sample_times = []
    final_result = None

    for run_idx in range(repeat_times):
        compute_metrics = (run_idx == 0)
        result = evaluate_once(args, dataset, compute_metrics=compute_metrics, istest=istest and compute_metrics)

        per_sample_times.append(result["per_sample_time"])

        if compute_metrics:
            final_result = result

        print(
            f"Run {run_idx + 1}/{repeat_times}: "
            f"total inference time = {result['total_time']:.6f} s, "
            f"per-sample inference time = {result['per_sample_time']:.9f} s"
        )

    mean_time = statistics.mean(per_sample_times)

    if len(per_sample_times) > 1:
        std_time = statistics.stdev(per_sample_times)  # sample std
        # 95% CI with n=5 => t_(0.975, df=4) = 2.776
        t_critical = 2.776
        ci_half = t_critical * std_time / math.sqrt(len(per_sample_times))
    else:
        std_time = 0.0
        ci_half = 0.0

    print("=" * 60)
    print("Per-sample inference latency (95% CI)")
    print(f"{mean_time:.4f} +/- {ci_half:.4f} s/sample")
    print(f"{mean_time * 1000:.4f} +/- {ci_half * 1000:.4f} ms/sample")
    print("=" * 60)

    return final_result, per_sample_times, mean_time, ci_half


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser, require_train_dev=False)
    tokenizer_opts(parser)

    parser.add_argument("--soft_targets", action="store_true", help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5, help="Weight of the soft targets loss.")
    parser.add_argument("--labels_num", type=int, help="labels_num")
    parser.add_argument("--project_name", type=str, default="DCS", help="name of project")
    parser.add_argument("--name", type=str, default="ID = 4", help="name of process")

    args = parser.parse_args()
    args = load_hyperparam(args)

    args.logger = init_logger(args)
    set_seed(args.seed)
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = Classifier(args)
    load_finetuned_checkpoint(model, args.output_model_path)
    model = model.to(args.device)

    args.model = model

    if args.test_path is not None:
        args.logger.info("Test set evaluation.")
        testset = read_dataset(args, args.test_path)
        evaluate_with_ci(args, testset, istest=True, repeat_times=5)
    elif args.dev_path is not None:
        args.logger.info("Dev set evaluation.")
        devset = read_dataset(args, args.dev_path)
        evaluate_with_ci(args, devset, istest=False, repeat_times=5)
    else:
        raise ValueError("Please provide --test_path or --dev_path.")


if __name__ == "__main__":
    main()

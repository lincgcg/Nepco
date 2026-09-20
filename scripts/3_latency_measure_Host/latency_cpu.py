"""
CPU-only test script for CNN classifier.
- Loop over batch sizes: 1,2,4,8,16,32,64,128,256,512,1024
- For each batch size:
    * warmup once
    * measure 5 times
    * metric = total inference time / number of test samples
    * report 95% CI as mean ± half-width
- Also report significance between adjacent batch sizes using 95% CI overlap.
"""

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


BATCH_SIZES_TO_TEST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


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
    Try to enable the fast inference cache path for the current CNN encoder.
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


def build_test_tensors(dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])
    return src, tgt, seg


def batch_loader(batch_size, src, tgt, seg):
    instances_num = src.size(0)
    for i in range(0, instances_num, batch_size):
        yield (
            src[i: i + batch_size],
            tgt[i: i + batch_size],
            seg[i: i + batch_size],
        )


def evaluate_metrics_once(args, src, tgt, seg, batch_size, istest=False):
    """
    Compute metrics once, outside the repeated latency measurements.
    """
    device = args.device
    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    with torch.inference_mode():
        for src_batch, tgt_batch, seg_batch in batch_loader(batch_size, src, tgt, seg):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            seg_batch = seg_batch.to(device)

            logits = args.model.infer(src_batch, seg_batch)
            pred = torch.argmax(logits, dim=1)
            gold = tgt_batch

            correct += torch.sum(pred == gold).item()

            pred_cpu = pred.cpu()
            gold_cpu = gold.cpu()
            indices = pred_cpu * args.labels_num + gold_cpu
            bincount = torch.bincount(indices, minlength=args.labels_num * args.labels_num)
            confusion += bincount.view(args.labels_num, args.labels_num)

    acc = correct / src.size(0)

    if istest:
        print(confusion)
        eps = 1e-9
        prf_dir = os.path.join(os.path.dirname(args.output_model_path), "prf")
        os.makedirs(prf_dir, exist_ok=True)
        filename = os.path.join(prf_dir, "prf.csv")

        with open(filename, "w+", encoding="utf-8") as f2:
            f2.write("Label num,Precision,Recall,F1\n")
            for i in range(confusion.size(0)):
                p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
                r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
                f1 = 2 * p * r / (p + r + eps)

                args.logger.info("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
                f2.write("{},{},{},{}\n".format(i, p, r, f1))

    args.logger.info("Acc. (Correct/Total): {:.4f} ({}/{})".format(acc, correct, src.size(0)))
    return acc, confusion


def evaluate_time_once(args, src, tgt, seg, batch_size):
    """
    Run one full timed inference pass.
    The measured time only covers args.model.infer(src_batch, seg_batch).
    """
    device = args.device
    total_time = 0.0

    args.model.eval()

    with torch.inference_mode():
        for src_batch, _, seg_batch in batch_loader(batch_size, src, tgt, seg):
            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)

            start_t = time.perf_counter()
            _ = args.model.infer(src_batch, seg_batch)
            end_t = time.perf_counter()

            total_time += (end_t - start_t)

    per_sample_time = total_time / src.size(0)
    return total_time, per_sample_time


def compute_95ci(values):
    mean_v = statistics.mean(values)
    if len(values) > 1:
        std_v = statistics.stdev(values)
        # n=5, df=4, 95% CI => t_(0.975, 4)=2.776
        t_critical = 2.776
        half_width = t_critical * std_v / math.sqrt(len(values))
    else:
        half_width = 0.0
    return mean_v, half_width


def ci_overlap(mean1, half1, mean2, half2):
    low1, high1 = mean1 - half1, mean1 + half1
    low2, high2 = mean2 - half2, mean2 + half2
    return not (high1 < low2 or high2 < low1)


def benchmark_batch_sizes(args, dataset, istest=False, repeat_times=5):
    src, tgt, seg = build_test_tensors(dataset)

    metric_bs = min(args.batch_size, len(dataset))
    args.logger.info(f"Computing classification metrics once with batch_size={metric_bs}.")
    evaluate_metrics_once(args, src, tgt, seg, metric_bs, istest=istest)

    results = []

    print("\n" + "=" * 80)
    print("Starting CPU batch-size significance analysis")
    print("=" * 80)

    for bs in BATCH_SIZES_TO_TEST:
        print(f"\n[Batch Size = {bs}]")

        # warmup
        _ = evaluate_time_once(args, src, tgt, seg, bs)

        per_sample_times = []
        total_times = []

        for run_idx in range(repeat_times):
            total_time, per_sample_time = evaluate_time_once(args, src, tgt, seg, bs)
            total_times.append(total_time)
            per_sample_times.append(per_sample_time)

            print(
                f"Run {run_idx + 1}/{repeat_times}: "
                f"total inference time = {total_time:.4f} s, "
                f"per-sample inference time = {per_sample_time * 1000:.4f} ms/sample"
            )

        mean_time, ci_half = compute_95ci(per_sample_times)

        result = {
            "batch_size": bs,
            "per_sample_times": per_sample_times,
            "mean_time": mean_time,
            "ci_half": ci_half,
        }
        results.append(result)

        print(f"95% CI: {mean_time * 1000:.4f} +/- {ci_half * 1000:.4f} ms/sample")

        if len(results) > 1:
            prev = results[-2]
            overlap = ci_overlap(prev["mean_time"], prev["ci_half"], mean_time, ci_half)

            if overlap:
                sig_msg = "no clear difference (95% CIs overlap)"
            else:
                sig_msg = "significant difference (95% CIs do not overlap)"

            improvement = (prev["mean_time"] - mean_time) / prev["mean_time"] * 100.0
            print(
                f"Compared with batch_size={prev['batch_size']}: {sig_msg}, "
                f"per-sample time change = {improvement:.4f}%"
            )

    print("\n" + "=" * 80)
    print("Summary (ms/sample, 95% CI)")
    print("=" * 80)
    for r in results:
        print(
            f"batch_size={r['batch_size']:>4d}: "
            f"{r['mean_time'] * 1000:.4f} +/- {r['ci_half'] * 1000:.4f} ms/sample"
        )

    return results


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser, require_train_dev=False)
    tokenizer_opts(parser)

    parser.add_argument("--soft_targets", action='store_true', help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5, help="Weight of the soft targets loss.")
    parser.add_argument("--labels_num", type=int, help="labels_num")
    parser.add_argument("--project_name", type=str, default="DCS", help="name of project")
    parser.add_argument("--name", type=str, default="ID = 4", help="name of process")

    args = parser.parse_args()
    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)
    set_seed(args.seed)
    args.logger = init_logger(args)

    args.device = torch.device("cpu")

    model = Classifier(args)
    load_finetuned_checkpoint(model, args.output_model_path)
    model = model.to(args.device)

    args.model = model
    prepare_model_for_fast_test(args.model)

    if args.test_path is not None:
        args.logger.info("CPU Test set evaluation.")
        testset = read_dataset(args, args.test_path)
        benchmark_batch_sizes(args, testset, istest=True, repeat_times=5)
    elif args.dev_path is not None:
        args.logger.info("CPU Dev set evaluation.")
        devset = read_dataset(args, args.dev_path)
        benchmark_batch_sizes(args, devset, istest=False, repeat_times=5)
    else:
        raise ValueError("Please provide --test_path or --dev_path.")


if __name__ == "__main__":
    main()

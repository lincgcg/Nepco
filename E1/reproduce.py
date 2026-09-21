#!/usr/bin/env python3
"""Reload a fine-tuned Nepco checkpoint and evaluate a labelled TSV file."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


E1_DIR = Path(__file__).resolve().parent
AE_ROOT = E1_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from uer.embeddings import Embedding, str2embedding  # noqa: E402
from uer.encoders import str2encoder  # noqa: E402
from uer.utils import str2tokenizer  # noqa: E402
from uer.utils.config import load_hyperparam  # noqa: E402
from uer.utils.constants import CLS_TOKEN, PAD_TOKEN, SEP_TOKEN  # noqa: E402
from uer.utils.misc import pooling  # noqa: E402


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = Embedding(args)
        for name in args.embedding:
            self.embedding.update(str2embedding[name](args, len(args.tokenizer.vocab)), name)
        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)

    def infer(self, src, seg):
        output = self.encoder(self.embedding(src, seg), seg)
        output = pooling(output, seg, self.pooling_type)
        return self.output_layer_2(torch.tanh(self.output_layer_1(output)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seq_length", default=128, type=int)
    parser.add_argument("--pooling", default="max")
    parser.add_argument("--embedding", nargs="+", default=["word"])
    parser.add_argument("--tokenizer", default="bert")
    parser.add_argument("--do_lower_case", default="true")
    parser.add_argument("--spm_model_path", default=None)
    parser.add_argument("--merges_path", default=None)
    parser.add_argument("--labels_num", required=True, type=int)
    parser.add_argument("--soft_targets", action="store_true")
    parser.add_argument("--soft_alpha", default=0.5, type=float)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--results_dir", default=str(E1_DIR / "results"))
    args = parser.parse_args()
    return load_hyperparam(args)


def load_checkpoint(model, path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must be a state_dict, got {type(state).__name__}")
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def read_dataset(args):
    rows = []
    pad_id = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
    with open(args.test_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"label", "text_a"}.issubset(reader.fieldnames):
            raise ValueError("TSV header must contain label and text_a")
        for row_id, row in enumerate(reader):
            label = int(row["label"])
            if not 0 <= label < args.labels_num:
                raise ValueError(f"row {row_id + 2}: label {label} is outside [0, {args.labels_num})")
            src = args.tokenizer.convert_tokens_to_ids(
                [CLS_TOKEN] + args.tokenizer.tokenize(row["text_a"]) + [SEP_TOKEN]
            )
            seg = [1] * len(src)
            src = src[: args.seq_length]
            seg = seg[: args.seq_length]
            if len(src) < args.seq_length:
                missing = args.seq_length - len(src)
                src.extend([pad_id] * missing)
                seg.extend([0] * missing)
            rows.append((src, label, seg))
            if args.max_samples is not None and len(rows) >= args.max_samples:
                break
    if not rows:
        raise ValueError("No samples were loaded from the TSV file")
    return rows


def batches(rows, batch_size):
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        yield (
            torch.tensor([item[0] for item in chunk], dtype=torch.long),
            torch.tensor([item[1] for item in chunk], dtype=torch.long),
            torch.tensor([item[2] for item in chunk], dtype=torch.long),
        )


def main():
    args = parse_args()
    for path in (args.output_model_path, args.test_path, args.config_path, args.vocab_path):
        if not path or not Path(path).is_file():
            raise FileNotFoundError(path)

    args.tokenizer = str2tokenizer[args.tokenizer](args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    model = Classifier(args)
    load_checkpoint(model, args.output_model_path)
    model.to(device).eval()
    rows = read_dataset(args)

    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    predictions = []
    with torch.inference_mode():
        for src, gold, seg in batches(rows, args.batch_size):
            pred = model.infer(src.to(device), seg.to(device)).argmax(dim=1).cpu()
            indices = pred * args.labels_num + gold
            confusion += torch.bincount(
                indices, minlength=args.labels_num * args.labels_num
            ).view(args.labels_num, args.labels_num)
            predictions.extend(zip(gold.tolist(), pred.tolist()))

    eps = 1e-12
    per_class = []
    for label in range(args.labels_num):
        tp = confusion[label, label].item()
        precision = tp / (confusion[label, :].sum().item() + eps)
        recall = tp / (confusion[:, label].sum().item() + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        per_class.append((label, precision, recall, f1))

    summary = {
        "samples": len(rows),
        "accuracy": sum(gold == pred for gold, pred in predictions) / len(rows),
        "precision_macro": sum(item[1] for item in per_class) / args.labels_num,
        "recall_macro": sum(item[2] for item in per_class) / args.labels_num,
        "f1_macro": sum(item[3] for item in per_class) / args.labels_num,
        "model_path": str(Path(args.output_model_path).resolve()),
        "test_path": str(Path(args.test_path).resolve()),
        "device": str(device),
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (results_dir / "prf.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("label", "precision", "recall", "f1"))
        writer.writerows(per_class)
    with (results_dir / "predictions.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("row_id", "label", "prediction"))
        for row_id, (gold, pred) in enumerate(predictions):
            writer.writerow((row_id, gold, pred))

    print(f"Precision: {summary['precision_macro']:.10f}")
    print(f"Recall:    {summary['recall_macro']:.10f}")
    print(f"Macro F1:  {summary['f1_macro']:.10f}")
    print(f"Accuracy:  {summary['accuracy']:.10f}")
    print(f"Samples:   {summary['samples']}")
    print(f"Results:   {results_dir.resolve()}")


if __name__ == "__main__":
    main()

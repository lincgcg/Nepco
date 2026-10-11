#!/usr/bin/env python3
"""Prepare class-balanced few-shot TSVs and sequentially fine-tune Nepco.

Adapted from the historical run_classifier_DCS_shot_LR_loaded_parallel.py
sampling logic: shuffle each class with one fixed RNG, take up to `shot`
examples per class, then shuffle the sampled training set. Validation and
test TSVs are passed unchanged to run_finetune.sh. No LR search is performed.

Example (run from the artifact root):
  CUDA_VISIBLE_DEVICES=0 python3 scripts/2_model_train/model_finetune/few_shot.py \
      --train_path data/train_dataset.tsv --dev_path data/valid_dataset.tsv \
      --test_path data/test_dataset.tsv --labels_num 15 \
      --shot_list 5 25 50 250 500 2500 5000 --learning_rate 5e-4

For a nominal reference of 5000 examples/class, these shot counts correspond
to 0.1%, 0.5%, 1%, 5%, 10%, 50%, 100%. They are NOT percentages of the supplied
training TSV. Classes with fewer examples use all available training examples;
sampling.json records requested and actual counts. Use --strict_shots to
reject undersupplied classes. Use --prepare_only to generate TSVs without
loading a model or starting training. Existing output directories are rejected.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import random
import subprocess


DEFAULT_SHOTS = [5, 25, 50, 250, 500, 2500, 5000]


def read_train(path, labels_num):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader, None)
        if not header or header.count("label") != 1 or "text_a" not in header:
            raise ValueError("Training TSV requires unique label and text_a columns.")
        label_column = header.index("label")
        rows = []
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError("Malformed training TSV row {}.".format(line_number))
            label = int(row[label_column])
            if not 0 <= label < labels_num:
                raise ValueError("Out-of-range label at row {}.".format(line_number))
            rows.append(row)
    counts = Counter(int(row[label_column]) for row in rows)
    if set(counts) != set(range(labels_num)):
        raise ValueError("Training TSV must contain every label in [0, labels_num).")
    return header, rows, label_column, counts


def sample_rows(rows, label_column, shot, seed):
    by_label = defaultdict(list)
    for row in rows:
        by_label[int(row[label_column])].append(row)
    rng = random.Random(seed)
    sampled = []
    for label in sorted(by_label):
        examples = list(by_label[label])
        rng.shuffle(examples)
        sampled.extend(examples[:shot])
    rng.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_path", type=Path, required=True)
    parser.add_argument("--dev_path", type=Path, required=True)
    parser.add_argument("--test_path", type=Path, required=True)
    parser.add_argument("--labels_num", type=int, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("outputs/few_shot"))
    parser.add_argument("--shot_list", type=int, nargs="+", default=DEFAULT_SHOTS)
    parser.add_argument("--shot_seed", type=int, default=3)
    parser.add_argument("--strict_shots", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--pretrained_model_path", type=Path,
                        default=Path("models/nepco_pretrained_model.bin"))
    parser.add_argument("--config_path", type=Path, default=Path("configs/nepco_config.json"))
    parser.add_argument("--vocab_path", type=Path, default=Path("vocab/hex_vocab.txt"))
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--epochs_num", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=128)
    args = parser.parse_args()
    if (args.labels_num <= 0 or any(shot <= 0 for shot in args.shot_list)
            or len(set(args.shot_list)) != len(args.shot_list)):
        parser.error("labels_num and shots must be positive; shots must be unique.")
    if (not 0 < args.learning_rate < float("inf") or args.epochs_num <= 0
            or args.batch_size <= 0 or args.seq_length <= 0):
        parser.error("Training hyperparameters must be positive and finite.")
    for path in (args.train_path, args.dev_path, args.test_path):
        if not path.is_file():
            parser.error("Missing input TSV: {}".format(path))
    header, rows, label_column, counts = read_train(args.train_path, args.labels_num)
    if args.strict_shots and min(counts.values()) < max(args.shot_list):
        parser.error("Some classes contain fewer training examples than requested.")
    if not args.prepare_only:
        for path in (args.pretrained_model_path, args.config_path, args.vocab_path):
            if not path.is_file():
                parser.error("Missing model input: {}".format(path))
    output_root = args.output_root.resolve()
    targets = [output_root / "shot_{}".format(shot) for shot in args.shot_list]
    if any(path.exists() for path in targets):
        parser.error("A shot output directory already exists; choose a new output_root.")
    runner = Path(__file__).resolve().with_name("run_finetune.sh")
    if not args.prepare_only and not runner.is_file():
        parser.error("run_finetune.sh must be next to this script.")
    # Resolve caller-relative paths before run_finetune.sh changes directory.
    inputs = {"DEV_PATH": str(args.dev_path.resolve()),
              "TEST_PATH": str(args.test_path.resolve()),
              "PRETRAINED_MODEL_PATH": str(args.pretrained_model_path.resolve()),
              "CONFIG_PATH": str(args.config_path.resolve()),
              "VOCAB_PATH": str(args.vocab_path.resolve())}
    for shot, target in zip(args.shot_list, targets):
        target.mkdir(parents=True, exist_ok=False)
        sampled = sample_rows(rows, label_column, shot, args.shot_seed)
        train_path = target / "train_dataset.tsv"
        with train_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(sampled)
        actual = Counter(int(row[label_column]) for row in sampled)
        metadata = {"requested_shots_per_class": shot, "shot_seed": args.shot_seed,
                    "source_train_counts": dict(sorted(counts.items())),
                    "sampled_train_counts": dict(sorted(actual.items())),
                    "sampled_train_total": len(sampled),
                    "validation_and_test": "unchanged; full supplied TSVs",
                    "learning_rate": args.learning_rate}
        with (target / "sampling.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
        print("shot={} train_total={} counts={}".format(shot, len(sampled), dict(actual)), flush=True)
        if args.prepare_only:
            continue
        env = os.environ.copy()
        env.update(inputs)
        env.update({"TRAIN_PATH": str(train_path), "LABELS_NUM": str(args.labels_num),
                    "OUTPUT_ROOT": str(target), "MODEL_DIR": str(target / "models"),
                    "PRF_DIR": str(target / "models/prf"),
                    "OUTPUT_MODEL_PATH": str(target / "models/finetuned_model.bin"),
                    "LEARNING_RATE": str(args.learning_rate), "EPOCHS_NUM": str(args.epochs_num),
                    "BATCH_SIZE": str(args.batch_size), "SEQ_LENGTH": str(args.seq_length),
                    "GPU_IDS": env.get("GPU_IDS", env.get("CUDA_VISIBLE_DEVICES", "0"))})
        with (target / "train.log").open("x", encoding="utf-8") as log:
            subprocess.run(["bash", str(runner)], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()

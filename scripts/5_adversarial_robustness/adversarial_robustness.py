#!/usr/bin/env python3
"""Run gradient-attribution adversarial robustness experiments for CIC-EVSE."""

import argparse
import csv
import hashlib
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import tqdm


AE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, AE_ROOT)

from uer.embeddings import *  # noqa: F401,F403
from uer.encoders import *  # noqa: F401,F403
from uer.model_saver import save_model
from uer.opts import adv_opts, finetune_opts, tokenizer_opts
from uer.utils import *  # noqa: F401,F403
from uer.utils.config import load_hyperparam
from uer.utils.constants import CLS_TOKEN, PAD_TOKEN, SEP_TOKEN
from uer.utils.logging import init_logger
from uer.utils.misc import pooling
from uer.utils.optimizers import *  # noqa: F401,F403
from uer.utils.seed import set_seed


SUMMARY_FIELDS = [
    "Dataset",
    "Model",
    "Run_Seed",
    "Attack_Seed",
    "Strategy",
    "Ratio",
    "Stage",
    "Train_Set",
    "Valid_Set",
    "Test_Set",
    "Init_From",
    "Epochs",
    "Learning_Rate",
    "Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Best_Dev_Accuracy",
    "Model_Path",
    "Test_Path",
]

@dataclass
class Example:
    index: int
    row: List[str]
    label: int
    text_a: str
    text_b: Optional[str]
    raw_tokens: List[str]
    src: List[int]
    seg: List[int]
    token_to_positions: List[List[int]]
    soft_tgt: Optional[List[float]] = None
    importance: Optional[List[float]] = None


@dataclass
class DatasetPack:
    split: str
    path: str
    header: List[str]
    columns: Dict[str, int]
    examples: List[Example]


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

    def forward(self, src, tgt, seg, soft_tgt=None):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = pooling(output, seg, self.pooling_type)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + (
                    1 - self.soft_alpha
                ) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        return None, logits


class ResultWriter:
    def __init__(self, summary_path: Path):
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_file = open(summary_path, "w", newline="", encoding="utf-8")
        self.summary_writer = csv.DictWriter(self.summary_file, fieldnames=SUMMARY_FIELDS)
        self.summary_writer.writeheader()

    def close(self):
        self.summary_file.close()

    def write(
        self,
        args,
        attack_seed: str,
        strategy: str,
        ratio: str,
        stage: str,
        train_set: str,
        valid_set: str,
        test_set: str,
        init_from: str,
        epochs: int,
        model_path: str,
        test_path: str,
        metrics: Dict[str, object],
        best_dev_accuracy: Optional[float] = None,
    ):
        row = {
            "Dataset": args.dataset_name,
            "Model": args.model_name,
            "Run_Seed": args.run_seed,
            "Attack_Seed": attack_seed,
            "Strategy": strategy,
            "Ratio": ratio,
            "Stage": stage,
            "Train_Set": train_set,
            "Valid_Set": valid_set,
            "Test_Set": test_set,
            "Init_From": init_from,
            "Epochs": epochs,
            "Learning_Rate": args.learning_rate,
            "Accuracy": metrics["accuracy"],
            "Macro_Precision": metrics["macro_precision"],
            "Macro_Recall": metrics["macro_recall"],
            "Macro_F1": metrics["macro_f1"],
            "Best_Dev_Accuracy": "" if best_dev_accuracy is None else best_dev_accuracy,
            "Model_Path": model_path,
            "Test_Path": test_path,
        }
        self.summary_writer.writerow(row)
        self.summary_file.flush()


def ratio_to_name(ratio: float) -> str:
    if float(ratio).is_integer():
        return str(int(ratio))
    return str(ratio).replace(".", "p")


def stable_int(*parts: object) -> int:
    key = "|".join(str(part) for part in parts)
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:16], 16)


def load_or_initialize_parameters(args, model, model_path: Optional[str] = None, strict: bool = False):
    load_path = model_path if model_path is not None else args.pretrained_model_path
    if load_path is not None:
        model.load_state_dict(torch.load(load_path, map_location="cpu"), strict=strict)
    else:
        for name, param in list(model.named_parameters()):
            if "gamma" not in name and "beta" not in name:
                param.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
        )
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, examples: Sequence[Example]):
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        src_batch = torch.LongTensor([sample.src for sample in batch])
        tgt_batch = torch.LongTensor([sample.label for sample in batch])
        seg_batch = torch.LongTensor([sample.seg for sample in batch])
        if batch and batch[0].soft_tgt is not None:
            soft_tgt_batch = torch.FloatTensor([sample.soft_tgt for sample in batch])
        else:
            soft_tgt_batch = None
        yield batch, src_batch, tgt_batch, seg_batch, soft_tgt_batch


def encode_text(args, text_a: str, text_b: Optional[str] = None) -> Tuple[List[int], List[int], List[List[int]], List[str]]:
    raw_tokens = text_a.strip().split()
    text_a_tokens: List[str] = []
    token_to_positions: List[List[int]] = []

    for raw_token in raw_tokens:
        sub_tokens = args.tokenizer.tokenize(raw_token)
        positions = []
        for sub_token in sub_tokens:
            positions.append(1 + len(text_a_tokens))
            text_a_tokens.append(sub_token)
        token_to_positions.append(positions)

    if text_b is None:
        tokens = [CLS_TOKEN] + text_a_tokens + [SEP_TOKEN]
        seg = [1] * len(tokens)
    else:
        text_b_tokens = args.tokenizer.tokenize(text_b)
        tokens_a = [CLS_TOKEN] + text_a_tokens + [SEP_TOKEN]
        tokens_b = text_b_tokens + [SEP_TOKEN]
        tokens = tokens_a + tokens_b
        seg = [1] * len(tokens_a) + [2] * len(tokens_b)

    if len(tokens) > args.seq_length:
        tokens = tokens[: args.seq_length]
        seg = seg[: args.seq_length]
    token_to_positions = [[pos for pos in positions if pos < len(tokens)] for positions in token_to_positions]

    src = args.tokenizer.convert_tokens_to_ids(tokens)
    pad_id = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
    while len(src) < args.seq_length:
        src.append(pad_id)
        seg.append(0)

    return src, seg, token_to_positions, raw_tokens


def read_dataset(args, path: str, split: str) -> DatasetPack:
    examples: List[Example] = []
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        columns = {column_name: i for i, column_name in enumerate(header)}
        if "label" not in columns or "text_a" not in columns:
            raise ValueError(f"{path} must contain label and text_a columns.")

        for row_id, row in enumerate(reader):
            if not row:
                continue
            while len(row) < len(header):
                row.append("")
            label = int(row[columns["label"]])
            text_a = row[columns["text_a"]]
            text_b = row[columns["text_b"]] if "text_b" in columns else None
            src, seg, token_to_positions, raw_tokens = encode_text(args, text_a, text_b)
            soft_tgt = None
            if args.soft_targets and "logits" in columns:
                soft_tgt = [float(value) for value in row[columns["logits"]].split(" ")]
            examples.append(
                Example(
                    index=row_id,
                    row=row,
                    label=label,
                    text_a=text_a,
                    text_b=text_b,
                    raw_tokens=raw_tokens,
                    src=src,
                    seg=seg,
                    token_to_positions=token_to_positions,
                    soft_tgt=soft_tgt,
                )
            )
    return DatasetPack(split=split, path=path, header=header, columns=columns, examples=examples)


def write_dataset(pack: DatasetPack, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(pack.header)
        for example in pack.examples:
            writer.writerow(example.row)


def build_model(args, model_path: Optional[str] = None, strict: bool = False) -> nn.Module:
    model = Classifier(args)
    load_or_initialize_parameters(args, model, model_path=model_path, strict=strict)
    model = model.to(args.device)
    return model


def train_batch(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()
    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    loss.backward()

    if args.use_adv and args.adv_type == "fgm":
        args.adv_method.attack(epsilon=args.fgm_epsilon)
        loss_adv, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
        loss_adv.backward()
        args.adv_method.restore()

    if args.use_adv and args.adv_type == "pgd":
        args.adv_method.backup_grad()
        for step in range(args.pgd_k):
            args.adv_method.attack(
                epsilon=args.pgd_epsilon,
                alpha=args.pgd_alpha,
                is_first_attack=(step == 0),
            )
            if step != args.pgd_k - 1:
                model.zero_grad()
            else:
                args.adv_method.restore_grad()
            loss_adv, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            loss_adv.backward()
        args.adv_method.restore()

    optimizer.step()
    scheduler.step()
    return loss


def evaluate(args, model, examples: Sequence[Example]) -> Dict[str, object]:
    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    model.eval()

    for _, src_batch, tgt_batch, seg_batch, _ in batch_loader(args.batch_size, examples):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = model(src_batch, None, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        for j in range(pred.size()[0]):
            confusion[pred[j].item(), tgt_batch[j].item()] += 1
        correct += torch.sum(pred == tgt_batch).item()

    eps = 1e-9
    precision_list = []
    recall_list = []
    f1_list = []
    for label_id in range(args.labels_num):
        precision = confusion[label_id, label_id].item() / (confusion[label_id, :].sum().item() + eps)
        recall = confusion[label_id, label_id].item() / (confusion[:, label_id].sum().item() + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "accuracy": correct / max(len(examples), 1),
        "macro_precision": sum(precision_list) / max(len(precision_list), 1),
        "macro_recall": sum(recall_list) / max(len(recall_list), 1),
        "macro_f1": sum(f1_list) / max(len(f1_list), 1),
        "confusion": confusion,
    }


def train_and_select(args, model, train_examples: List[Example], dev_examples: List[Example], model_path: Path) -> float:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    args.train_steps = int(len(train_examples) * args.epochs_num / args.batch_size) + 1
    optimizer, scheduler = build_optimizer(args, model)
    if args.use_adv:
        args.adv_method = str2adv[args.adv_type](model)

    best_dev_accuracy = -1.0
    total_loss = 0.0
    args.logger.info("Start training: {} examples, {} epochs.".format(len(train_examples), args.epochs_num))
    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        epoch_examples = list(train_examples)
        random.shuffle(epoch_examples)
        model.train()
        for step, (_, src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(
            batch_loader(args.batch_size, epoch_examples),
            start=1,
        ):
            loss = train_batch(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            total_loss += loss.item()
            if step % args.report_steps == 0:
                args.logger.info(
                    "Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(
                        epoch, step, total_loss / args.report_steps
                    )
                )
                total_loss = 0.0

        dev_metrics = evaluate(args, model, dev_examples)
        dev_accuracy = dev_metrics["accuracy"]
        args.logger.info("Epoch id: {}, Dev acc: {:.4f}".format(epoch, dev_accuracy))
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            save_model(model, str(model_path))

    model.load_state_dict(torch.load(str(model_path), map_location=args.device))
    return best_dev_accuracy


def reduce_subtoken_importance(args, token_scores: torch.Tensor, example: Example) -> List[float]:
    values = token_scores.detach().cpu().tolist()
    reduced = []
    for positions in example.token_to_positions:
        valid_positions = [pos for pos in positions if pos < len(values) and example.seg[pos] != 0]
        if not valid_positions:
            reduced.append(0.0)
            continue
        selected = [values[pos] for pos in valid_positions]
        if args.importance_reduce == "sum":
            reduced.append(sum(selected))
        elif args.importance_reduce == "max":
            reduced.append(max(selected))
        else:
            reduced.append(sum(selected) / len(selected))
    return reduced


def compute_importance(args, model, examples: List[Example], split: str):
    captured = {}

    def capture_embedding(_, __, output):
        captured["embedding"] = output
        output.retain_grad()

    handle = model.embedding.register_forward_hook(capture_embedding)
    model.eval()
    args.logger.info("Computing gradient attribution for {} set.".format(split))

    try:
        for batch_examples, src_batch, tgt_batch, seg_batch, _ in tqdm.tqdm(
            batch_loader(args.batch_size, examples),
            total=math.ceil(len(examples) / args.batch_size),
        ):
            model.zero_grad()
            captured.clear()
            src_batch = src_batch.to(args.device)
            tgt_batch = tgt_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)

            _, logits = model(src_batch, None, seg_batch)
            if args.importance_target == "gold":
                targets = tgt_batch
            else:
                targets = torch.argmax(logits.detach(), dim=1)
            selected_logits = logits.gather(1, targets.view(-1, 1)).sum()
            selected_logits.backward()

            embedding = captured["embedding"]
            token_scores = (embedding.grad.detach() * embedding.detach()).abs().sum(dim=-1)
            for row_id, example in enumerate(batch_examples):
                example.importance = reduce_subtoken_importance(args, token_scores[row_id], example)
    finally:
        handle.remove()
        model.zero_grad()


def save_importance(pack: DatasetPack, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sample_id", "label", "token_index", "token", "importance"])
        for example in pack.examples:
            if example.importance is None:
                continue
            for token_index, token in enumerate(example.raw_tokens):
                writer.writerow([example.index, example.label, token_index, token, example.importance[token_index]])


def is_hex_token(token: str) -> bool:
    if token == "":
        return False
    return all(ch in "0123456789abcdefABCDEF" for ch in token)


def random_hex_like(token: str, rng: random.Random) -> str:
    width = len(token)
    alphabet = "0123456789abcdef"
    for _ in range(16):
        candidate = "".join(rng.choice(alphabet) for _ in range(width))
        if candidate.lower() != token.lower():
            return candidate
    return "".join(rng.choice(alphabet) for _ in range(width))


def replacement_token(args, old_token: str, rng: random.Random, observed_tokens: Sequence[str]) -> str:
    if args.replacement_mode == "observed" or not is_hex_token(old_token):
        if not observed_tokens:
            return old_token
        for _ in range(16):
            candidate = observed_tokens[rng.randrange(len(observed_tokens))]
            if candidate != old_token:
                return candidate
        return observed_tokens[rng.randrange(len(observed_tokens))]
    return random_hex_like(old_token, rng)


def attack_indices(example: Example, ratio: float, strategy: str, attack_seed: int, split: str) -> List[int]:
    candidates = [i for i, positions in enumerate(example.token_to_positions) if positions]
    if ratio <= 0 or not candidates:
        return []
    k = max(1, int(math.ceil(len(candidates) * ratio / 100.0)))
    k = min(k, len(candidates))
    if strategy == "Top":
        if example.importance is None:
            raise ValueError("Top attack requires importance scores.")
        return sorted(
            candidates,
            key=lambda idx: (-example.importance[idx], idx),
        )[:k]

    rng = random.Random(stable_int("select", attack_seed, split, example.index, ratio, strategy))
    return sorted(rng.sample(candidates, k))


def build_attacked_pack(
    args,
    pack: DatasetPack,
    strategy: str,
    ratio: float,
    attack_seed: int,
    observed_tokens: Sequence[str],
) -> DatasetPack:
    examples = []
    text_a_col = pack.columns["text_a"]
    text_b_col = pack.columns.get("text_b")
    ratio_name = ratio_to_name(ratio)

    for example in pack.examples:
        tokens = list(example.raw_tokens)
        selected_indices = attack_indices(example, ratio, strategy, attack_seed, pack.split)
        for token_index in selected_indices:
            rng = random.Random(stable_int("replace", attack_seed, pack.split, example.index, ratio_name, token_index))
            tokens[token_index] = replacement_token(args, tokens[token_index], rng, observed_tokens)

        row = list(example.row)
        row[text_a_col] = " ".join(tokens)
        text_b = row[text_b_col] if text_b_col is not None else None
        src, seg, token_to_positions, raw_tokens = encode_text(args, row[text_a_col], text_b)
        attacked = Example(
            index=example.index,
            row=row,
            label=example.label,
            text_a=row[text_a_col],
            text_b=text_b,
            raw_tokens=raw_tokens,
            src=src,
            seg=seg,
            token_to_positions=token_to_positions,
            soft_tgt=example.soft_tgt,
        )
        examples.append(attacked)

    return DatasetPack(
        split=pack.split,
        path=pack.path,
        header=pack.header,
        columns=pack.columns,
        examples=examples,
    )


def collect_observed_tokens(packs: Iterable[DatasetPack]) -> List[str]:
    tokens = []
    for pack in packs:
        for example in pack.examples:
            tokens.extend(example.raw_tokens)
    return sorted(set(tokens))


def format_set_name(strategy: str, ratio: float) -> str:
    if strategy == "Clean":
        return "Clean"
    return "{}-{}".format(strategy, ratio_to_name(ratio))


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_experiment(args):
    output_root = Path(args.output_root)
    model_root = output_root / args.dataset_name / args.model_name / args.run_seed
    clean_model_path = model_root / "models" / "Clean" / "finetuned_model.bin"
    results_dir = model_root / "results"

    args.logger.info("Reading clean train/dev/test TSV files once.")
    train_pack = read_dataset(args, args.train_path, "train")
    dev_pack = read_dataset(args, args.dev_path, "valid")
    test_pack = read_dataset(args, args.test_path, "test")
    observed_tokens = collect_observed_tokens([train_pack, dev_pack, test_pack])

    set_seed(args.seed)
    args.logger.info("Clean fine-tuning starts.")
    clean_model = build_model(args)
    clean_best_dev = train_and_select(args, clean_model, train_pack.examples, dev_pack.examples, clean_model_path)
    clean_metrics = evaluate(args, clean_model, test_pack.examples)

    for pack in [train_pack, dev_pack, test_pack]:
        compute_importance(args, clean_model, pack.examples, pack.split)
        if args.save_importance:
            save_importance(pack, model_root / "importance" / f"{pack.split}_importance.tsv")

    for attack_seed in args.attack_seeds:
        summary_path = results_dir / f"adv_summary_AttackSeed_{attack_seed}.csv"
        writer = ResultWriter(summary_path)
        try:
            writer.write(
                args=args,
                attack_seed=str(attack_seed),
                strategy="Clean",
                ratio="0",
                stage="Clean_Finetune",
                train_set="Clean",
                valid_set="Clean",
                test_set="Clean",
                init_from="Pretrained",
                epochs=args.epochs_num,
                model_path=str(clean_model_path),
                test_path=args.test_path,
                metrics=clean_metrics,
                best_dev_accuracy=clean_best_dev,
            )

            for ratio in args.ratios:
                ratio_name = ratio_to_name(ratio)
                for strategy in ["Top", "Random"]:
                    args.logger.info(
                        "Building attacked datasets: strategy={}, ratio={}, attack_seed={}.".format(
                            strategy, ratio_name, attack_seed
                        )
                    )
                    attacked_train = build_attacked_pack(args, train_pack, strategy, ratio, attack_seed, observed_tokens)
                    attacked_dev = build_attacked_pack(args, dev_pack, strategy, ratio, attack_seed, observed_tokens)
                    attacked_test = build_attacked_pack(args, test_pack, strategy, ratio, attack_seed, observed_tokens)

                    attacked_data_dir = (
                        model_root
                        / "datasets"
                        / strategy
                        / f"Ratio_{ratio_name}"
                        / f"AttackSeed_{attack_seed}"
                    )
                    if args.save_attacked_tsv:
                        write_dataset(attacked_train, attacked_data_dir / "train_dataset.tsv")
                        write_dataset(attacked_dev, attacked_data_dir / "valid_dataset.tsv")
                        write_dataset(attacked_test, attacked_data_dir / "test_dataset.tsv")
                        attacked_test_path = str(attacked_data_dir / "test_dataset.tsv")
                    else:
                        attacked_test_path = "in_memory"

                    fixed_metrics = evaluate(args, clean_model, attacked_test.examples)
                    writer.write(
                        args=args,
                        attack_seed=str(attack_seed),
                        strategy=strategy,
                        ratio=ratio_name,
                        stage="Fixed_Attack_Test",
                        train_set="Clean",
                        valid_set="Clean",
                        test_set=format_set_name(strategy, ratio),
                        init_from="Clean_Checkpoint",
                        epochs=0,
                        model_path=str(clean_model_path),
                        test_path=attacked_test_path,
                        metrics=fixed_metrics,
                        best_dev_accuracy=clean_best_dev,
                    )

                    clean_model.to("cpu")
                    clear_cuda_cache()

                    set_seed(args.seed)
                    if args.attack_aware_init == "clean":
                        init_path = str(clean_model_path)
                        init_from = "Clean_Checkpoint"
                        strict = True
                    else:
                        init_path = None
                        init_from = "Pretrained"
                        strict = False

                    attack_model = build_model(args, model_path=init_path, strict=strict)
                    attack_model_path = (
                        model_root
                        / "models"
                        / strategy
                        / f"Ratio_{ratio_name}"
                        / f"AttackSeed_{attack_seed}"
                        / "finetuned_model.bin"
                    )
                    attack_best_dev = train_and_select(
                        args,
                        attack_model,
                        attacked_train.examples,
                        attacked_dev.examples,
                        attack_model_path,
                    )
                    attack_metrics = evaluate(args, attack_model, attacked_test.examples)
                    writer.write(
                        args=args,
                        attack_seed=str(attack_seed),
                        strategy=strategy,
                        ratio=ratio_name,
                        stage="Attack_Aware_Finetune",
                        train_set=format_set_name(strategy, ratio),
                        valid_set=format_set_name(strategy, ratio),
                        test_set=format_set_name(strategy, ratio),
                        init_from=init_from,
                        epochs=args.epochs_num,
                        model_path=str(attack_model_path),
                        test_path=attacked_test_path,
                        metrics=attack_metrics,
                        best_dev_accuracy=attack_best_dev,
                    )
                    del attack_model
                    clear_cuda_cache()
                    clean_model.to(args.device)
            args.logger.info("Attack-seed CSV: {}".format(summary_path))
        finally:
            writer.close()

    args.logger.info("Finished writing per-attack-seed summary CSV files in {}.".format(results_dir))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)
    parser.set_defaults(learning_rate=5e-4)
    tokenizer_opts(parser)
    parser.add_argument("--soft_targets", action="store_true", help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5, help="Weight of the soft targets loss.")
    parser.add_argument("--labels_num", type=int, required=True, help="Number of labels.")
    parser.add_argument("--project_name", type=str, default="DCS-Adv-Robustness", help="Project name.")
    parser.add_argument("--name", type=str, default="adv_robustness", help="Run name.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name written to output CSV files.")
    parser.add_argument("--dataset_name", type=str, default="CIC-EVSE", help="Dataset name written to output CSV files.")
    parser.add_argument("--run_seed", type=str, default="01", help="Dataset split seed label written to output CSV files.")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory for models, attacked TSVs, and CSVs.")
    parser.add_argument("--ratios", type=float, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--attack_seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--importance_target", choices=["predicted", "gold"], default="predicted")
    parser.add_argument("--importance_reduce", choices=["mean", "sum", "max"], default="mean")
    parser.add_argument("--replacement_mode", choices=["hex", "observed"], default="hex")
    parser.add_argument("--attack_aware_init", choices=["pretrained", "clean"], default="pretrained")
    parser.add_argument("--save_attacked_tsv", action="store_true")
    parser.add_argument("--save_importance", action="store_true")
    adv_opts(parser)
    args = parser.parse_args()
    args = load_hyperparam(args)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    args.logger = init_logger(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.logger.info("Device: {}".format(args.device))
    args.logger.info("Model: {}, dataset: {}, run seed: {}".format(args.model_name, args.dataset_name, args.run_seed))
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    return args


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

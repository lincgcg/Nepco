#!/usr/bin/env python3
"""Evaluate packaged TraGe models on full E1 test TSVs; never train."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

E1 = Path(__file__).resolve().parent
RUNTIME = E1 / "TraGe"
sys.path.insert(0, str(RUNTIME))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def evaluate(dataset, device="cpu", results_dir=None, verify=True):
    import torch
    from argparse import Namespace
    from classifier import Classifier
    from dataset import read_dataset
    from uer.utils.tokenizers import BertTokenizer

    torch.set_num_threads(2)
    artifact = E1 / "artifacts" / dataset / "TraGe"
    config = json.loads((artifact / "model_config.json").read_text())
    test_path = E1 / config["test_path"]
    model_path = artifact / "finetuned_model.bin"
    if verify:
        manifest = json.loads((RUNTIME / "manifest.json").read_text())
        if manifest["status"] != "complete_verified":
            raise RuntimeError("TraGe package has not passed final verification")
        records = dict(manifest["runtime_sha256"])
        records.update(manifest["datasets"][dataset]["file_sha256"])
        for relative, digest in records.items():
            if sha(E1 / relative) != digest:
                raise RuntimeError("Checksum mismatch: " + relative)
    if sha(test_path) != config["test_sha256"]:
        raise RuntimeError("Accepted test TSV changed")
    args = Namespace(**config["model"])
    args.vocab_path = str(RUNTIME / "vocab/encryptd_vocab.txt")
    args.spm_model_path = None
    args.do_lower_case = "true"
    args.tokenizer = BertTokenizer(args)

    model = Classifier(args)
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    target_device = torch.device(device)
    model.to(target_device).eval()
    rows = read_dataset(args, test_path)
    if len(rows) != config["test_samples"]:
        raise ValueError("Full test-set sample count mismatch")
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            src = torch.tensor([r[0] for r in batch], dtype=torch.long, device=target_device)
            seg = torch.tensor([r[2] for r in batch], dtype=torch.long, device=target_device)
            pred = torch.softmax(model(src, seg), dim=1).argmax(dim=1).cpu()
            gold = torch.tensor([r[1] for r in batch], dtype=torch.long)
            confusion += torch.bincount(pred * args.labels_num + gold,
                minlength=args.labels_num ** 2).reshape(args.labels_num, args.labels_num)
            predictions.extend(zip(gold.tolist(), pred.tolist()))
    per_class = []
    for label in range(args.labels_num):
        tp = confusion[label, label].item()
        p = tp / (confusion[label].sum().item() + 1e-9)
        r = tp / (confusion[:, label].sum().item() + 1e-9)
        per_class.append((label, p, r, 2*p*r/(p+r+1e-9)))
    metrics = {"dataset": dataset, "method": "TraGe", "samples": len(rows),
        "accuracy": confusion.diag().sum().item()/len(rows),
        "precision_macro": sum(r[1] for r in per_class)/args.labels_num,
        "recall_macro": sum(r[2] for r in per_class)/args.labels_num,
        "f1_macro": sum(r[3] for r in per_class)/args.labels_num,
        "confusion": confusion.tolist(), "prf_epsilon": 1e-9}
    if verify:
        expected = manifest["datasets"][dataset]["metrics"]
        if confusion.tolist() != expected["confusion"]:
            raise RuntimeError("Reloaded predictions differ from verified package")
    output = Path(results_dir) if results_dir else E1/"results"/dataset/"TraGe"/"reproduced"
    output.mkdir(parents=True, exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    with (output/"prf.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("label", "precision", "recall", "f1"))
        w.writerows(per_class)
    with (output/"predictions.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(("row_id", "label", "prediction"))
        w.writerows((i, g, p) for i, (g, p) in enumerate(predictions))
    print(f"{dataset}: Precision={metrics['precision_macro']:.10f} "
          f"Recall={metrics['recall_macro']:.10f} Macro F1={metrics['f1_macro']:.10f}", flush=True)
    return metrics

def main():
    manifest = json.loads((RUNTIME/"manifest.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["all"]+list(manifest["datasets"]), nargs="?", default="DataCon")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--results_dir")
    args = parser.parse_args()
    if args.dataset == "all" and args.results_dir:
        parser.error("--results_dir requires one dataset")
    for name in (manifest["datasets"] if args.dataset == "all" else [args.dataset]):
        evaluate(name, args.device, args.results_dir)

if __name__ == "__main__":
    main()

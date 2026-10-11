# E1: DataCon smoke test

This directory provides a quick experiment on DataCon and comprehensive experiments on all nine datasets, including Nepco and TraGe.
The scripts reload the supplied fine-tuned models and evaluate their test TSV files; they do not retrain models.

Before running either model, download the approximately 5 GB artifact release from Google Drive and place the dataset folders under `E1/artifacts/` as described in [Download E1 artifacts](artifacts/README.md).

## Quick experiment: DataCon

### Nepco

Run from the Nepco artifact root:

```bash
bash E1/run_smoke_test.sh
```

This evaluates the full DataCon test set.

## Comprehensive experiment: all nine datasets

### Nepco

```bash
bash E1/run_smoke_test.sh --all
```

CPU is the default. Add `--device cuda` to the all-dataset command to use CUDA. To evaluate one other dataset:

```bash
python3 E1/reproduce_dataset.py USTC-TFC --device cpu
```

The all-dataset entry point checks artifact hashes and runs all nine full test sets. Fresh results are saved under `results/<dataset>/reproduced/`, preserving accepted results in `results/<dataset>/`.

## Files and configurations

### Nepco

- `artifacts/<dataset>/`: `finetuned_model.bin`, `test_dataset.tsv`, and `model_config.json`.
- `results/<dataset>/`: reference `metrics.json`, `prf.csv`, and `predictions.tsv`.
- `manifest.json`: configurations, verified metrics, and SHA-256 checksums for all nine datasets.
- `reproduce.py`: shared model-reload and evaluation implementation.
- `run_smoke_test.sh`: original DataCon entry point, extended with `--all`.

Nepco inference uses sequence length 128, batch size 32, max pooling, and word embeddings. Each dataset's architecture is supplied in `model_config.json`. CSTNET-120 uses the 96-byte data variant; USTC-TFC uses nonoverlap bigram stride2 and packet prefix128. No training or validation data is required.

## TraGe

TraGe uses a separate inference-only Transformer runtime, vocabulary, and reproduction entry point. It shares the existing test TSVs with Nepco; no pretraining checkpoint or train/validation TSVs are needed for inference.

Run from the Nepco artifact root, as with the Nepco commands above:

```bash
# Quick experiment: full DataCon test set
bash E1/run_trage.sh

# Comprehensive experiment: all nine full test sets
bash E1/run_trage.sh all --device cuda

# One selected dataset
bash E1/run_trage.sh USTC-TFC --device cuda
```

CPU is the default; omit `--device cuda` to run on CPU. Each command checks package hashes and exact agreement with the verified confusion matrix. Fresh results are saved under `E1/results/<dataset>/TraGe/reproduced/`, preserving accepted results in `E1/results/<dataset>/TraGe/`.

- `artifacts/<dataset>/TraGe/`: `finetuned_model.bin` and `model_config.json`.
- `artifacts/<dataset>/test_dataset.tsv`: the existing test TSV shared with Nepco.
- `results/<dataset>/TraGe/`: reference `metrics.json`, `prf.csv`, and `predictions.tsv`.
- `TraGe/manifest.json`: verified metrics, expected confusion matrices, and SHA-256 checksums.
- `TraGe/classifier.py`, `TraGe/dataset.py`: inference model and full-test TSV reader.
- `TraGe/uer/`, `TraGe/vocab/`, `TraGe/special_tokens_map.json`: required inference dependencies and TraGe vocabulary.
- `reproduce_trage.py`, `run_trage.sh`: model-reload and full-test entry points; no training.

TraGe inference uses sequence length 128, batch size 32, and first-token pooling. Each dataset's `model_config.json` supplies the model architecture and test-set metadata. CSTNET-120 uses the 96-byte input variant; USTC-TFC uses nonoverlap payload-128.

TraGe inference dependencies: Python 3, PyTorch, packaging, regex, and six. Install the dependencies in your Python environment before running the commands. The shell scripts use `python3` from the current environment by default; no specific Conda environment or installation path is required.

## Verified full-test results

### Nepco

The verified full-test result is:

| Dataset | Precision | Recall | Macro F1 |
|---|---:|---:|---:|
| DataCon | 0.9920384649 | 0.9912944099 | 0.9916037393 |
| CIC-EVSE | 0.8871187098 | 0.8904631132 | 0.8824131676 |
| CIC-IoMT | 0.8343574138 | 0.8080725965 | 0.8160454268 |
| ISCX-VPN-App | 0.7038003592 | 0.7070583514 | 0.7033850763 |
| ISCX-VPN-Service | 0.9252022275 | 0.9188670129 | 0.9209466015 |
| USTC-TFC | 0.9664221029 | 0.9640000000 | 0.9638322397 |
| CSTNET-20 | 0.8881866150 | 0.8746239140 | 0.8714466582 |
| CSTNET-120 | 0.6739587276 | 0.6481867627 | 0.6437680810 |
| DoHBrw | 0.6392464248 | 0.6308000000 | 0.6333643499 |

Precision and Recall are unweighted macro averages over the configured classes. These are independently verified results for supplied fixed configurations, not test-score maxima from a search.

### TraGe

The verified full-test result is:

| Dataset | Precision | Recall | Macro F1 |
|---|---:|---:|---:|
| DataCon | 0.9836845676 | 0.9922260869 | 0.9877266856 |
| CIC-EVSE | 0.8322583673 | 0.8432186687 | 0.8349310710 |
| CIC-IoMT | 0.8659966206 | 0.8150610960 | 0.8164106455 |
| ISCX-VPN-App | 0.7209536452 | 0.7122047178 | 0.7118848017 |
| ISCX-VPN-Service | 0.9257616659 | 0.9151952509 | 0.9190349585 |
| USTC-TFC | 0.9600867898 | 0.9582000000 | 0.9581058835 |
| CSTNET-20 | 0.8737496747 | 0.8811148342 | 0.8677341093 |
| CSTNET-120 | 0.6753287444 | 0.6478146324 | 0.6383783507 |
| DoHBrw | 0.6235771835 | 0.6148000000 | 0.6172041520 |

# E1: DataCon smoke test

This directory turns the selected DataCon result into a reload-only smoke test.
It does not retrain during reproduction. The two input artifacts are:

```text
artifacts/DataCon/finetuned_model.bin
artifacts/DataCon/test_dataset.tsv
```

Run the full test set on CPU:

```bash
PYTHON_BIN=python3 bash E1/run_smoke_test.sh
```

The command prints macro Precision, Recall, Macro F1, Accuracy, and sample count.
It also writes `metrics.json`, `prf.csv`, and `predictions.tsv` under
`E1/results/DataCon/`.

For a quicker code-path check, append `--max_samples 32`. The verified result
below uses all 2,857 test samples without `--max_samples`. Its locked provenance
is DataCon split seed 11, model seed 7, pre-training learning rate 1e-3,
fine-tuning learning rate 5e-3, sequence length 128, and 10 epochs.

The verified full-test result is:

```text
Precision: 0.9920384649
Recall:    0.9912944099
Macro F1:  0.9916037393
Accuracy:  0.9866993350
Samples:   2857
```

These values fall inside the reported intervals. `manifest.json` records the
configuration and SHA-256 checksums for the two reload inputs.

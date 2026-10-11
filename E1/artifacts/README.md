# Download E1 Artifacts

The fine-tuned models and test datasets total approximately 5 GB. Because of their size, they are distributed separately through Google Drive rather than stored directly in this Git repository.

Download the artifacts from the following shared Google Drive folder:

**[Download E1 artifacts (Google Drive)](https://drive.google.com/drive/folders/1UafVUJeugAGDHBASkYFizduFhkPXuiBF?usp=drive_link)**

The folder link is the distribution location for this release. No machine-specific server path is required. If Google Drive requests access, please contact the repository maintainers to check the sharing permissions.

## Contents

The release contains artifacts for all nine datasets:

- DataCon
- CIC-EVSE
- CIC-IoMT
- ISCX-VPN-App
- ISCX-VPN-Service
- USTC-TFC
- CSTNET-20
- CSTNET-120
- DoHBrw

Each dataset directory contains:

```text
<dataset>/
├── finetuned_model.bin
├── model_config.json
├── test_dataset.tsv
└── TraGe/
    ├── finetuned_model.bin
    └── model_config.json
```

The top-level `finetuned_model.bin` and `model_config.json` are the Nepco checkpoint and architecture configuration. The files under `TraGe/` are the corresponding TraGe checkpoint and configuration. `test_dataset.tsv` is the accepted labeled test set shared by the two models. These files support inference only; no training or validation datasets are needed for E1.

## Installation

1. Download all nine dataset folders from the link above. Google Drive may package the download into multiple ZIP archives; download and extract all parts before continuing.
2. Copy the nine dataset folders directly into this repository's `E1/artifacts/` directory, preserving their names and internal structure.
3. Check that the resulting layout is:

   ```text
   E1/artifacts/
   ├── README.md
   ├── DataCon/
   ├── CIC-EVSE/
   ├── CIC-IoMT/
   ├── ISCX-VPN-App/
   ├── ISCX-VPN-Service/
   ├── USTC-TFC/
   ├── CSTNET-20/
   ├── CSTNET-120/
   └── DoHBrw/
   ```

For example, the DataCon files must be located at `E1/artifacts/DataCon/finetuned_model.bin`, `E1/artifacts/DataCon/test_dataset.tsv`, and `E1/artifacts/DataCon/TraGe/finetuned_model.bin`.

Do not introduce an extra `artifacts/` or download-folder level, such as `E1/artifacts/artifacts/DataCon/`. Do not rename or modify the downloaded files; the E1 reproduction entry points verify their SHA-256 checksums against the packaged manifests.

Once the files are in place, run the existing commands from the repository root:

```bash
# Full DataCon test set on CPU.
bash E1/run_smoke_test.sh
bash E1/run_trage.sh

# All nine full test sets on CPU.
bash E1/run_smoke_test.sh --all
bash E1/run_trage.sh all
```

See [E1/README.md](../README.md) for evaluation details and verified results. E2 also reuses the DataCon models and test TSV from this directory; no second copy is needed.

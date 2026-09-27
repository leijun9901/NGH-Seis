# Quick start

## Clone and install

```bash
git clone https://github.com/leijun9901/NGH-Seis.git
cd NGH-Seis
conda env create -f environment.yml
conda activate ngh-seis
```

## Synthetic code check

This command does not require the dataset:

```bash
python scripts/smoke_test.py --synthetic --model all
```

Validation status: pending final clean-environment reproducibility audit.

## Obtain the dataset

Download NGH-Seis v1.0 from Science Data Bank:

https://doi.org/10.57760/sciencedb.013j7

Extract the release under `outputs/NGH-Seis_v1.0/` or keep it elsewhere and pass its `metadata` directory through `--release`.

## Inspect one released sample

```bash
python examples/inspect_ngh_seis_sample.py \
  --candidate-id 1 \
  --production-root outputs/NGH-Seis_v1.0 \
  --release-root outputs/NGH-Seis_v1.0/metadata
```

Candidate identifiers can contain gaps, so choose a filename that exists in the downloaded manifest.

## Start an IID benchmark

```bash
python scripts/train_ngh_seis_task1.py \
  --release outputs/NGH-Seis_v1.0/metadata \
  --model unet \
  --output outputs/task1_iid_unet
```

Use the corresponding `train_ngh_seis_task2.py` entry point for Task 2. See `docs/REPRODUCIBILITY.md` before attempting full or structural-OOD runs.

# Reproducibility plan

Validation status: pending final clean-environment reproducibility audit.

This document defines the checks planned after the paper-release documentation update is approved. The commands below are not recorded as passed by this preparation commit.

## Level 0 — synthetic smoke test

Purpose: verify installation, model construction and one forward/backward update without downloading the dataset.

```bash
python scripts/smoke_test.py --synthetic --model all
```

Planned evidence: clean environment specification, command log, package versions and machine/CPU or GPU details.

## Level 1 — real Science Data Bank data loading

Purpose: verify anonymous access, extracted layout, release metadata and one-realization loading for both tasks.

```bash
python scripts/verify_scientific_data_release.py \
  /path/to/NGH-Seis_v1.0 \
  /path/to/NGH-Seis_v1.0/metadata/accepted_manifest.json

python scripts/smoke_test.py /path/to/NGH-Seis_v1.0 --model all
```

Planned checks include the real directory tree, manifest consistency, quantitative RTM selection, mask handling, target shapes and saturation units.

## Level 2 — reference benchmark reproduction

Purpose: reproduce the documented IID and structural-OOD benchmark protocols using the frozen splits and training-only calibrations.

```bash
python scripts/train_ngh_seis_task1.py \
  --release /path/to/NGH-Seis_v1.0/metadata \
  --model unet \
  --output outputs/task1_iid_unet

python scripts/train_ngh_seis_task2.py \
  --release /path/to/NGH-Seis_v1.0/metadata \
  --model unet \
  --output outputs/task2_iid_unet
```

The same process is planned for `resunet` and `deeplabv3plus`. Structural-OOD runs add `--protocol structural-ood --ood-fold FOLD_NAME` and use a distinct output directory.

## Frozen conventions

- Task 1 inputs: `input_rtm_conditioned_unscaled` and `migration_Vp`.
- Task 1 targets: `Vp` update and log acoustic-impedance residual using training-only calibration.
- Task 2 input: `input_rtm_conditioned_unscaled` only.
- Task 2 targets: `Sh` and `Sg` fractions.
- Model selection uses validation loss; test partitions are not used for calibration or selection.
- `valid_mask` and `loss_mask` are metadata/evaluation masks, not input channels.

Do not interpret this plan as a completed reproducibility claim.

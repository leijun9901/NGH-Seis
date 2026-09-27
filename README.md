# NGH-Seis v1.0

[![Smoke test](https://github.com/leijun9901/NGH-Seis/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/leijun9901/NGH-Seis/actions/workflows/smoke-test.yml)

NGH-Seis v1.0 is a physics-based two-dimensional synthetic marine seismic dataset for acoustic-property reconstruction and hydrate/free-gas characterization. It contains 1,000 realizations balanced across five geological scenario families.

## Quick links

- Dataset: [Science Data Bank V1](https://doi.org/10.57760/sciencedb.013j7)
- Code: [GitHub repository](https://github.com/leijun9901/NGH-Seis)
- Dataset citation: [DOI 10.57760/sciencedb.013j7](https://doi.org/10.57760/sciencedb.013j7)
- Code license: [MIT](LICENSE)
- Data license: [CC BY 4.0](DATA_LICENSE.md)
- Paper: Scientific Data Data Descriptor; citation will be added after publication

## Data and code

**Dataset.** NGH-Seis v1.0 is archived in Science Data Bank:

https://doi.org/10.57760/sciencedb.013j7

**Code.** This repository contains the generation, quality-control, loading and reference-benchmark code associated with NGH-Seis v1.0. The approximately 45 GB dataset is not stored in GitHub.

**Licenses.** Code is released under the MIT License. Dataset files in Science Data Bank are licensed under CC BY 4.0.

## What is NGH-Seis?

The release provides co-registered geological properties, 25-shot marine seismic observations, quantitative reverse-time-migration (RTM) inputs and benchmark targets. The 1,000 accepted realizations comprise 200 samples from each of five families. The fixed IID split is 800/100/100, and each of the five structural out-of-distribution folds is 720/80/200.

## Dataset contents

Each realization links four files:

1. geological-property archive;
2. 25-shot observation archive;
3. training-pair archive;
4. RTM quality report.

The full release contains 4,000 data files totaling approximately 45.044 GB (41.951 GiB). After extraction, use this layout:

```text
NGH-Seis_v1.0/
├── data/
│   ├── geology/
│   ├── observations/
│   └── training_pairs/
├── quality/
│   └── rtm_reports/
├── configs/
└── metadata/
```

See [Data access](docs/DATA_ACCESS.md), [Data layout](docs/DATA_LAYOUT.md) and the detailed [data dictionary](metadata/DATA_DICTIONARY.md).

## Repository contents

- `configs/`: frozen geology and acquisition configurations;
- `scripts/`: generation, quality control, release verification, smoke test and benchmark entry points;
- `src/datasets/`: Task 1 and Task 2 loaders;
- `src/models/`: U-Net, ResUNet and DeepLabV3+ reference implementations;
- `metadata/`: lightweight schema, split definitions and training-only calibration metadata;
- `examples/`: minimal inspection examples;
- `docs/`: data access, layout, quick-start and reproducibility guidance.

## Installation

```bash
conda env create -f environment.yml
conda activate ngh-seis
```

Alternatively, install the pinned Python requirements in a compatible environment:

```bash
pip install -r requirements.txt
```

## Data access

Download NGH-Seis v1.0 only from Science Data Bank and extract it as `outputs/NGH-Seis_v1.0/`, or choose another location and pass its `metadata` directory through `--release`.

```bash
python scripts/verify_scientific_data_release.py \
  outputs/NGH-Seis_v1.0 \
  outputs/NGH-Seis_v1.0/metadata/accepted_manifest.json
```

This command is documented for the later reproducibility audit; the paper-release documentation update itself does not claim that a fresh download has already been validated. See [DATA_ACCESS.md](docs/DATA_ACCESS.md).

## Quick code check

The synthetic smoke test does not require the published dataset:

```bash
python scripts/smoke_test.py --synthetic --model all
```

Validation status: pending final clean-environment reproducibility audit.

## Loading the dataset

Pass the downloaded release metadata directory to the loaders or training scripts. For example:

```bash
python scripts/train_ngh_seis_task1.py \
  --release outputs/NGH-Seis_v1.0/metadata \
  --model unet \
  --output outputs/task1_iid_unet
```

The quantitative modelling field is `input_rtm_conditioned_unscaled`. `input_rtm_display_only` is for visualization only. `valid_mask` is support metadata and `loss_mask` defines the optimization/evaluation region; neither mask is an input channel.

Archived `Sh` and `Sg` values are dimensionless pore-volume fractions. Figures may display percentages. The Dice thresholds `Sh >= 0.01` and `Sg >= 0.003` correspond to 1% and 0.3%, respectively.

## Reference benchmark tasks

- **Task 1:** quantitative RTM plus smooth migration `Vp` as inputs; predict a `Vp` update and a log acoustic-impedance residual relative to the training-only log-linear baseline.
- **Task 2:** quantitative RTM only; predict hydrate saturation `Sh` and free-gas saturation `Sg`.
- **Networks:** U-Net, ResUNet and DeepLabV3+.

Full IID training entry points are:

```bash
python scripts/train_ngh_seis_task1.py --model unet
python scripts/train_ngh_seis_task2.py --model unet
```

Use `--protocol structural-ood --ood-fold FOLD_NAME` for a prescribed held-out-family fold. See [Reproducibility](docs/REPRODUCIBILITY.md) before interpreting results.

## Reproducibility

The repository separates three verification levels: synthetic installation smoke test, real Science Data Bank sample loading, and reference benchmark reproduction. This paper-release preparation does not mark those stages as completed. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Version mapping

- Dataset: NGH-Seis v1.0
- Data archive: Science Data Bank V1
- Data DOI: `10.57760/sciencedb.013j7`
- Code release: pending final paper-release tag
- Paper: Scientific Data Data Descriptor

## Citation

### Dataset

Lei, J., Guo, Z. & Liu, J. NGH-Seis v1.0: A physics-based synthetic marine seismic dataset for gas hydrate and free-gas characterization. Version V1, Science Data Bank (2026). https://doi.org/10.57760/sciencedb.013j7

### Paper

Paper citation will be added after publication.

### Code

Until a paper-release tag is published, cite this repository URL and the exact commit used. Do not use the dataset DOI as a code DOI. Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## Licenses

- Code in this repository: MIT License.
- NGH-Seis v1.0 data hosted by Science Data Bank: CC BY 4.0.

## Troubleshooting

Start with [QUICKSTART.md](docs/QUICKSTART.md). Confirm that the downloaded root contains `data/`, `quality/`, `configs/` and `metadata/`, and pass the `metadata/` directory—not the repository root—to `--release`.

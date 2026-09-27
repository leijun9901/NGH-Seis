# Data access

## Authoritative archive

NGH-Seis v1.0 is publicly available from Science Data Bank:

- Version: V1
- DOI: https://doi.org/10.57760/sciencedb.013j7
- CSTR: 31253.11.sciencedb.013j7
- Data license: CC BY 4.0
- Approximate size: 45.044 GB (41.951 GiB)

Science Data Bank is the authoritative dataset source. GitHub contains code and lightweight metadata only.

## Expected extracted layout

Extract the downloaded archives into one common release root:

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

The full release contains 1,000 realizations and four linked files per realization, for 4,000 data files. The four files share the same four-digit `candidate_id`.

## Recommended placement

For repository-default commands, place the extracted root at:

```text
NGH-Seis/
└── outputs/
    └── NGH-Seis_v1.0/
```

The benchmark scripts accept another location through `--release`. Supply the path to the downloaded release's `metadata` directory:

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

The loaders infer the sibling `data/` directory from the supplied metadata path. Do not point `--release` at the GitHub `metadata/` mirror when running against real data; the loader needs the downloaded data files beside the downloaded metadata.

## Verification command reserved for the next audit stage

```bash
python scripts/verify_scientific_data_release.py \
  /path/to/NGH-Seis_v1.0 \
  /path/to/NGH-Seis_v1.0/metadata/accepted_manifest.json
```

Validation status: pending final clean-environment reproducibility audit. No full Science Data Bank download or checksum verification was performed during the repository-documentation stage.

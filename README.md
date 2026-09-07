# NGH-Seis

NGH-Seis v1.0 is a physics-based two-dimensional synthetic marine seismic
dataset for acoustic-property reconstruction and hydrate and free-gas saturation
estimation. It contains 1,000 realizations from five geological scenario
families, with 200 realizations per family.

This repository contains the code, frozen configurations, tests, and examples.
The 45.045 GB dataset is distributed through a separate Zenodo data record.

## Dataset contents

Each realization provides four files containing geological properties, 25-shot
marine streamer observations, co-registered benchmark arrays, and an RTM quality
report. The full release contains 4,000 data files. The public view has a shape
of 512 x 96, including 95 physical traces and one padding column.

Task 1 uses the conditioned RTM image and smooth migration velocity to estimate
P-wave velocity and acoustic impedance. Task 2 uses the conditioned RTM image
to estimate hydrate and free-gas saturation. Fixed IID and leave-one-family-out
partitions are included with the dataset.

## Installation

Create the documented Conda environment:

```bash
conda env create -f environment.yml
conda activate ngh-seis
```

Alternatively, install the Python requirements in an existing Python 3.12
environment:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Download and extract the Zenodo archives into a common `NGH-Seis-v1.0`
directory, then run:

```bash
python examples/inspect_ngh_seis_sample.py /path/to/NGH-Seis-v1.0 --candidate-id 1
```

The notebook `examples/NGH_Seis_quickstart.ipynb` provides the same loading and
visualization workflow interactively.

## Verification

The dataset record includes per-file SHA-256 digests. After extraction, run:

```bash
python scripts/verify_scientific_data_release.py \
  /path/to/NGH-Seis-v1.0 \
  /path/to/NGH-Seis-v1.0/metadata/accepted_manifest.json \
  --mode sha256
```

## Reproducing the dataset

The frozen generator configurations are:

- `configs/marine_streamer.json`
- `configs/ngh_seis_geology.json`

All public code, configuration files, metadata, and examples correspond to the
NGH-Seis v1.0 release.

## Citation and persistent identifiers

The code repository is `https://github.com/leijun9901/NGH-Seis`. The
version-specific data DOI, software DOI, and Data Descriptor DOI will be added
after publication of the corresponding records.

## License

Copyright (c) 2026 Central South University. The code is released under the MIT
License; see `LICENSE`. NGH-Seis dataset arrays, metadata, configurations, and
dataset documentation are released separately under the Creative Commons
Attribution 4.0 International License (CC BY 4.0) in the Zenodo record.

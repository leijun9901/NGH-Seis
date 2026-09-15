# NGH-Seis

NGH-Seis is a physics-based 2-D synthetic marine seismic dataset for acoustic-property reconstruction and hydrate/free-gas saturation estimation. The release contains 1,000 realizations from five geological families, with fixed IID and leave-one-family-out OOD splits.

The dataset is distributed separately through Zenodo (DOI pending). Extract it as `outputs/NGH-Seis_v1.0/`.

## Environment

```bash
conda env create -f environment.yml
conda activate ngh-seis
```

## Reproduce

Task 1 predicts P-wave velocity and acoustic impedance; Task 2 predicts hydrate and free-gas saturation. Available baselines are U-Net, ResUNet, and DeepLabV3+.

```bash
python scripts/smoke_test.py outputs/NGH-Seis_v1.0
```

The command reads a released sample and performs one forward/backward update for both tasks. A successful run prints `"status": "PASS"`.

Full training uses:

```bash
python scripts/train_ngh_seis_task1.py --model unet
python scripts/train_ngh_seis_task2.py --model unet
```

The frozen geological and acquisition configurations are in `configs/`. Dataset verification is provided by `scripts/verify_scientific_data_release.py`.

## Citation and license

Citation metadata is provided in `CITATION.cff`. The code is released under the MIT License.

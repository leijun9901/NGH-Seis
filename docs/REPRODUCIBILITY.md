# Reproducibility

Verification is separated into installation, data integrity, checkpoint evaluation and fresh training. Passing one stage does not imply that the others have passed.

The reviewer-revision audit verified all 4,000 files in the authors' frozen local release against the SHA-256 manifest. This was not a fresh full download from Science Data Bank. Evaluation of the original IID DeepLabV3+ checkpoints reproduced all recorded test metrics exactly on CUDA (PyTorch 2.5.1, RTX 4070). Public checkpoints and a hash-verified, complete fixed IID test subset are distributed with the [versioned release](https://github.com/leijun9901/NGH-Seis/releases/tag/v1.0.0); see [the one-command evaluation](REPORTED_RESULT_REPRODUCTION.md).

One fresh random-initialization 100-epoch DeepLabV3+ run per task also completed with the original settings and locally verified full dataset. Task 1 Vp/AI MAE changed from 42.87268/0.111304 to 43.22837/0.112979 (m/s and MRayl). Task 2 Sh/Sg Dice changed from 0.870144/0.772780 to 0.869419/0.766617. Original results remain unchanged. This single rerun is not bitwise identical and does not establish statistical equivalence or seed robustness. Exact commands, environment and all differences are in `metadata/fresh_training_reproduction.json`.

## Level 0 — synthetic smoke test

Purpose: verify installation, model construction and one forward/backward update without downloading the dataset.

```bash
python scripts/smoke_test.py --synthetic --model all
```

This checks execution only; it does not validate trained model accuracy.

The revision audit installed `requirements.txt` in a new Windows Python 3.12 virtual environment, passed `pip check`, and passed this command for all three models. The same environment also passed the released-data smoke test on realization 0017 for both tasks (one forward/backward/update per model and task). This does not establish training determinism, a full remote download or GitHub Actions status.

## Level 1 — real Science Data Bank data loading

Purpose: verify anonymous access, extracted layout, release metadata and one-realization loading for both tasks.

```bash
python scripts/verify_scientific_data_release.py \
  /path/to/NGH-Seis_v1.0 \
  /path/to/NGH-Seis_v1.0/metadata/accepted_manifest.json

python scripts/smoke_test.py /path/to/NGH-Seis_v1.0 --model all
```

Add `--mode sha256` to verify the complete contents rather than only file sizes. The loaders use the quantitative RTM, the frozen masks and the archived saturation fractions.

## Level 2 — reference benchmark reproduction

Purpose: reproduce the documented IID and structural-OOD benchmark protocols using the frozen splits and training-only calibrations.

```bash
python scripts/train_ngh_seis_task1.py \
  --release /path/to/NGH-Seis_v1.0/metadata \
  --model unet \
  --learning-rate 0.0008 --epochs 100 --batch-size 8 --seed 20260824 --amp \
  --output outputs/task1_iid_unet

python scripts/train_ngh_seis_task2.py \
  --release /path/to/NGH-Seis_v1.0/metadata \
  --model unet \
  --learning-rate 0.0008 --epochs 100 --batch-size 8 --seed 20260824 --amp \
  --output outputs/task2_iid_unet
```

For IID `resunet` and `deeplabv3plus`, change the model and use `--learning-rate 0.001`. Every structural-OOD run, including U-Net, uses `--learning-rate 0.001` together with `--protocol structural-ood --ood-fold FOLD_NAME` and a distinct output directory. Both IID U-Net runs used 0.0008; the other 34 formal runs used 0.001. These are the original settings, not a uniformly tuned architecture comparison. The script default of 0.0008 alone is insufficient to reproduce the other configurations.

The formal results use the default masked Smooth L1 loss. Keep the original loss when reproducing them. The seed, AMP option and learning rate should be explicit in saved commands.

## Position-wise mean baseline

```bash
python scripts/position_template_baseline.py \
  --release /path/to/NGH-Seis_v1.0/metadata \
  --methods mean --output outputs/position_mean.json
```

This fits each spatial template using only the relevant training partition and evaluates IID plus all five structural-OOD folds. It does not fit templates on test samples.

## Absolute-target input ablation

```bash
python scripts/run_task1_input_ablation.py \
  --release /path/to/NGH-Seis_v1.0/metadata \
  --mode rtm_plus_smooth_v --learning-rate 0.0008 \
  --epochs 100 --batch-size 8 --seed 20260824 --amp \
  --output outputs/ablation_rtm_plus_smooth_v
```

Repeat with `smooth_v_only` and `rtm_only`. The missing channel is zeroed and both outputs predict standardized absolute targets, as in the manuscript ablation; this is distinct from the residual-target formal Task 1 benchmark.

## Evaluate an archived checkpoint

```bash
python scripts/evaluate_checkpoint.py \
  --release /path/to/NGH-Seis_v1.0/metadata --task 1 \
  --checkpoint /path/to/task1_iid_deeplabv3plus/best.pt \
  --expected /path/to/task1_iid_deeplabv3plus/result.json \
  --output outputs/task1_checkpoint_evaluation.json
```

Use `--task 2` with the corresponding Task 2 checkpoint and result file. The two original IID DeepLabV3+ checkpoints are public release assets, separate from the Science Data Bank dataset archive. Checkpoint evaluation and random-initialization training are distinct checks.

## Frozen conventions

- Task 1 inputs: `input_rtm_conditioned_unscaled` and `migration_Vp`.
- Task 1 targets: `Vp` update and log acoustic-impedance residual using training-only calibration.
- Task 2 input: `input_rtm_conditioned_unscaled` only.
- Task 2 targets: `Sh` and `Sg` fractions.
- Model selection uses validation loss; test partitions are not used for calibration or selection.
- `valid_mask` and `loss_mask` are metadata/evaluation masks, not input channels.

The fixed structural-OOD folds test geological-family shift within this acoustic simulator. They do not test noise, acquisition changes or synthetic-to-field transfer.

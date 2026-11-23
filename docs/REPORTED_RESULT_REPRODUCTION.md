# Reproduce reported IID results

The versioned GitHub release contains both original IID U-Net checkpoints and evaluation fields for the complete fixed 100-realization IID test split. Each derived NPZ contains only fields needed by the unchanged loaders; array values, dtypes and shapes are preserved exactly. The manifest distinguishes derived-container hashes, original source-file hashes and array-level hashes. Frozen release metadata retain their original contents and original full-file hashes. This CC BY 4.0 test-only subset does not contain the 800-sample training partition or complete original geology/observation archives.

```bash
pip install -r requirements.txt
python scripts/reproduce_reported_results.py --device cpu
```

The command downloads approximately 127 MB, verifies every asset by SHA-256, verifies the extracted files and IID test IDs, evaluates both original checkpoints on all 100 test samples, and compares the main metrics at manuscript table precision. Downloads use `.partial` files and can resume when the server supports Range. Repeating the command reuses valid assets.

Use `--device cuda` for CUDA AMP evaluation in the reference configuration (PyTorch 2.5.1, CUDA 12.4, RTX 4070). CPU evaluation uses FP32. Both exact metric differences and rounded table comparisons are written to `outputs/reproduction/reported_result_reproduction.json`; numerical differences are not hidden. The process fails if a main metric does not match at the declared table precision.

Reference main metrics are Vp MAE 49.79 m/s and AI MAE 0.132 MRayl for Task 1, and Sh MAE 0.00244, Sh Dice 0.872, Sg MAE 0.000401 and Sg Dice 0.784 for Task 2. These describe the supplied original U-Net checkpoints, trained with initial learning rate 0.0008. Fresh DeepLabV3+ training is documented separately and need not reproduce weights or every decimal when GPU operations are not deterministic.

For an integrity-only check, add `--verify-only`. To evaluate one task, use `--task 1` or `--task 2`. To use manually downloaded assets, pass `--assets-dir /path/to/release_assets`; the same hashes are still checked.

The GitHub Actions workflow **Reported reference results** executes this CPU workflow and retains its JSON reports. The smaller **Smoke test** workflow checks model execution without data; it does not reproduce trained accuracy.

For training, download the complete dataset from [Science Data Bank](https://doi.org/10.57760/sciencedb.013j7), extract it into a common root, and run the complete-release verifier and training commands in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The full verifier expects all four files per realization and must not be used to claim that this test-only bundle is a complete release.

No user account, private file path, API token or author checkpoint directory is required for the GitHub test-bundle workflow. Original checkpoints were exported with tensor values unchanged and author-specific path arguments removed; source and public-checkpoint hashes are recorded.

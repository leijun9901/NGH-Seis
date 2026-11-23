# Reproduce reported IID results

The versioned GitHub release contains both original DeepLabV3+ checkpoints and the complete fixed IID test subset. The test bundle contains 100 original pair archives, their 100 original geology archives and frozen release metadata. It is a test-only redistribution of Science Data Bank V1 under CC BY 4.0; the NPZ files are byte-identical to the originals. It does not contain the 800-sample training partition.

```bash
pip install -r requirements.txt
python scripts/reproduce_reported_results.py --device cpu
```

The command downloads approximately 1.55 GB, verifies every asset by SHA-256, verifies the extracted files and IID test IDs, evaluates both original checkpoints on all 100 test samples, and compares the main metrics at the precision used in manuscript Tables 4–6. Downloads use `.partial` files and can resume when the server supports Range. Repeating the command reuses valid assets.

Use `--device cuda` for CUDA AMP evaluation in the reference configuration (PyTorch 2.5.1, CUDA 12.4, RTX 4070). CPU evaluation uses FP32. Both exact metric differences and rounded table comparisons are written to `outputs/reproduction/reported_result_reproduction.json`; numerical differences are not hidden. The process fails if a main metric does not match at the declared table precision.

Reference main metrics are Vp MAE 42.87 m/s and AI MAE 0.111 MRayl for Task 1, and Sh MAE 0.00230, Sh Dice 0.870, Sg MAE 0.000400 and Sg Dice 0.773 for Task 2. These describe the supplied original checkpoints. Fresh training is a separate experiment and need not reproduce weights or every decimal when GPU operations are not deterministic.

For an integrity-only check, add `--verify-only`. To evaluate one task, use `--task 1` or `--task 2`. To use manually downloaded assets, pass `--assets-dir /path/to/release_assets`; the same hashes are still checked.

The GitHub Actions workflow **Reported reference results** executes this CPU workflow and retains its JSON reports. The smaller **Smoke test** workflow checks model execution without data; it does not reproduce trained accuracy.

For training, download the complete dataset from [Science Data Bank](https://doi.org/10.57760/sciencedb.013j7), extract it into a common root, and run the complete-release verifier and training commands in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The full verifier expects all four files per realization and must not be used to claim that this test-only bundle is a complete release.

No user account, private file path, API token or author checkpoint directory is required for the GitHub test-bundle workflow. Original checkpoints were exported with tensor values unchanged and author-specific path arguments removed; source and public-checkpoint hashes are recorded.

# NGH-Seis v1.0 lightweight previews

The two `.npz` files in this directory let users inspect the release-view shapes and field names before downloading the full 41.951-GiB record.

- `candidate_0001_preview.npz`: diffuse continuous BSR example.
- `candidate_0004_preview.npz`: fault–gas-chimney example.

These files are **preview only**. `input_rtm_display_only` has per-sample display normalization and must not be used as the quantitative benchmark input. The complete release provides `input_rtm_conditioned_unscaled`, global training-only calibration, full 25-shot observations, full-grid geology, reports and checksums.

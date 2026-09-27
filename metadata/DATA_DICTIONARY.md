# NGH-Seis v1.0 data dictionary

## Scope

NGH-Seis v1.0 contains 1,000 accepted two-dimensional synthetic marine seismic realizations. The release is balanced across five geological scenario families (200 realizations per family). Every realization links full-grid geological and petrophysical fields, 25-shot marine-streamer observations, an independently migrated RTM-type image, and co-registered benchmark targets.

All NumPy archives use C-order arrays. For spatial fields the first axis is vertical/depth and the second is horizontal/distance. For gathers the order is shot, time, receiver.

## Public directory layout

```text
NGH-Seis_v1.0/
  data/
    geology/          1,000 geology and petrophysics archives
    observations/     1,000 full 25-shot observation archives
    training_pairs/   1,000 co-registered benchmark archives
  quality/
    rtm_reports/      1,000 per-realization RTM quality reports
  configs/            frozen geology and acquisition configurations
  metadata/           manifests, splits, calibration, schema and checksums
```

## File naming and identity

The four files belonging to one realization share the same four-digit `candidate_id`, for example `sample_0001_*`. This identifier is the immutable deterministic sample identity retained in filenames and manifests. Public `candidate_id` values need not be continuous. `release_index` is the continuous 1–1,000 ordering of the published release and should be used when a sequential index is required.

## Geological archive

File pattern: `data/geology/sample_NNNN_blake_geology.npz`

| Field | Shape | Type | Unit/definition | Role |
|---|---:|---|---|---|
| `x_m` | 1229 | float32 | m | Full-grid horizontal coordinate |
| `z_m` | 1001 | float32 | m below sea surface | Full-grid vertical coordinate |
| `seafloor_m` | 1229 | float32 | m below sea surface | Synthetic bathymetry |
| `bsr_m` | 1229 | float32 | m below sea surface | Nominal BSR surface |
| `phi_full` | 1001×1229 | float32 | fraction | Full-grid porosity |
| `Sh_full` | 1001×1229 | float32 | fraction of total pore volume | Full-grid hydrate saturation |
| `Sg_full` | 1001×1229 | float32 | fraction of total pore volume | Full-grid free-gas saturation |
| `Vp_full` | 1001×1229 | float32 | m s⁻¹ | Full-grid P-wave velocity |
| `rho_full` | 1001×1229 | float32 | kg m⁻³ | Full-grid bulk density |
| `lithology_proxy_full` | 1001×1229 | float32 | dimensionless proxy | Stochastic facies/lithology control |
| `phi_view` | 512×96 | float32 | fraction | Co-registered porosity view |
| `Sh_view` | 512×96 | float32 | fraction of total pore volume | Co-registered hydrate target |
| `Sg_view` | 512×96 | float32 | fraction of total pore volume | Co-registered free-gas target |
| `Vp_view` | 512×96 | float32 | m s⁻¹ | Co-registered Task 1 target |
| `rho_view` | 512×96 | float32 | kg m⁻³ | Co-registered density view |
| `AI_view` | 512×96 | float32 | kg m⁻² s⁻¹ | Co-registered Task 1 target |
| `lithology_view` | 512×96 | float32 | dimensionless proxy | Co-registered facies proxy |
| `valid_mask` | 512×96 | uint8 | 0 or 1 | Spatial support metadata |
| `loss_mask` | 512×96 | uint8 | 0 or 1 | Loss/evaluation support; not an input |
| `metadata_json` | scalar | Unicode | JSON | Per-realization generation metadata |

The full grid spacing is 5 m in both directions. The released view is sampled in a common seafloor-relative frame with 3 m vertical spacing and 37.5 m horizontal spacing. It contains 95 physical lateral traces plus one zero-valued padding column. Its nominal vertical range is −192 to 1,341 m relative to the seafloor; row 64 is the seafloor.

`AI_view` is obtained by anti-aliasing and resampling the full-grid impedance field. Because `AI`, `Vp`, and density are independently resampled, `AI_view` is not expected to equal the product of the two resampled arrays at machine precision. All released samples pass an NRMSE threshold of 0.1% and a 99.9th-percentile relative-error threshold of 1%.

## Observation archive

File pattern: `data/observations/sample_NNNN_observations25.npz`

| Field | Shape | Type | Unit/definition | Role |
|---|---:|---|---|---|
| `raw_gathers` | 25×1501×128 | float32 | solver pressure amplitude | Unconditioned simulated observations |
| `processed_gathers` | 25×1501×128 | float32 | relative amplitude | Fixed label-independent conditioning |
| `runtime_s` | scalar | float64 | s | Forward-model runtime |
| `peak_cuda_memory_gb` | scalar | float64 | GiB | Recorded peak CUDA tensor memory |

There are 25 shots, 1,501 time samples from 0 to 6.000 s inclusive, and 128 pressure receivers per shot. Receiver sampling is 4 ms. Shot spacing is 150 m, receiver spacing is 15 m, and source–receiver offsets range from 100 to 2,005 m. The amplitudes are numerical solver units and must not be interpreted as calibrated field pressure in pascals.

## Training-pair archive

File pattern: `data/training_pairs/sample_NNNN_training_pair25.npz`

| Field | Shape | Type | Unit/definition | Benchmark role |
|---|---:|---|---|---|
| `input_rtm_conditioned_unscaled` | 512×96 | float32 | adjoint-gradient amplitude | Primary quantitative RTM input for Tasks 1 and 2 |
| `input_rtm_display_only` | 512×96 | float32 | clipped display scale | Visualization only; never a quantitative input |
| `input_display_scale` | scalar | float64 | adjoint-gradient amplitude | Per-image display metadata only |
| `target_Sh` | 512×96 | float32 | pore-volume fraction | Task 2 target |
| `target_Sg` | 512×96 | float32 | pore-volume fraction | Task 2 target |
| `auxiliary_Vp` | 512×96 | float32 | m s⁻¹ | Task 1 true-Vp target |
| `migration_Vp` | 512×96 | float32 | m s⁻¹ | Task 1 second input and physical baseline |
| `valid_mask` | 512×96 | uint8 | 0 or 1 | Support metadata; not an input |
| `loss_mask` | 512×96 | uint8 | 0 or 1 | Loss/evaluation mask; not an input |
| `raw_representative_gather` | 1501×128 | float32 | solver pressure amplitude | Raw shot 13 for visualization |
| `processed_representative_gather` | 1501×128 | float32 | relative amplitude | Processed shot 13 for visualization |
| `shot_x_m` | 25 | float32 | m | Source horizontal coordinates |
| `receiver_x_m` | 25×128 | float32 | m | Receiver horizontal coordinates |

Forty-nine regularly sampled realizations also contain `diagnostic_true_model_rtm_conditioned_unscaled`, `diagnostic_true_model_rtm_display`, and `diagnostic_display_scale`. These fields are label-aware technical diagnostics and must not be used as benchmark inputs.

## Benchmark tasks

- Task 1, acoustic-property reconstruction: inputs are `input_rtm_conditioned_unscaled` and `migration_Vp`; targets are `auxiliary_Vp` and `AI_view`.
- Task 2, hydrate/free-gas saturation estimation: input is `input_rtm_conditioned_unscaled`; targets are `target_Sh` and `target_Sg`.

`loss_mask` is applied only when computing loss and metrics. It is not a network input. The released quantitative RTM input is not normalized independently per realization. Use the training-only scale in `rtm_amplitude_calibration.json`; the frozen 99.5th-percentile scale is 23,492.95703125 with a clipping multiple of 4.

## Splits and metadata

- `iid_splits_80_10_10.json`: stratified 800/100/100 train/validation/test split, with 160/20/20 realizations from each family.
- `structural_ood_leave_one_style_out.json`: five leave-one-family-out folds, each containing 720/80/200 train/validation/test realizations.
- `task1_acoustic_calibration.json`: Task 1 statistics fitted using only the IID training subset.
- `accepted_manifest.json` and `scientific_data_manifest.jsonl`: public file inventory and per-file metadata.
- `NGH_SEIS_ARRAY_SCHEMA.json`: machine-readable shapes, dtypes, units, roles, and observed extrema.
- `checksums.sha256`: SHA-256 digest for all 4,000 data files.

## Saturation conventions and bounds

Hydrate saturation `Sh` and free-gas saturation `Sg` are fractions of total pore volume, not percentages of bulk rock volume. Residual brine saturation is `Sb = 1 − Sh − Sg`. The accepted release satisfies `0 ≤ Sh ≤ 0.22`, `0 ≤ Sg ≤ 0.065`, and `Sh + Sg ≤ 0.25`. Multiply `Sh` or `Sg` by 100 only for percentage display.

## Important limitations

The dataset is a two-dimensional variable-density acoustic benchmark. It excludes elastic conversions, intrinsic attenuation, a pressure-release free surface, source/receiver ghosts, free-surface multiples, added field noise, and three-dimensional propagation. Blake Ridge and ODP Leg 164 information supplies regional prior ranges; the realizations are not a digital twin or site-specific inversion of EW0008 or ODP Site 997.

For download placement and loader paths, see `docs/DATA_ACCESS.md`. For a concise archive/key summary, see `docs/DATA_LAYOUT.md`.

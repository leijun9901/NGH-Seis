# Data layout

NGH-Seis v1.0 contains four linked files per realization. Spatial arrays use vertical/depth × horizontal/distance order. Gather arrays use shot × time × receiver order. The machine-readable authority for keys, shapes and dtypes is `metadata/NGH_SEIS_ARRAY_SCHEMA.json`; the complete field reference is `metadata/DATA_DICTIONARY.md`.

## Identity and filenames

- `candidate_id`: immutable deterministic generation identity, formatted with four digits in filenames; gaps are intentional.
- `release_index`: continuous public ordering from 1 to 1,000.

| Archive | Filename pattern | Purpose |
|---|---|---|
| Geological properties | `data/geology/sample_NNNN_blake_geology.npz` | Full-grid and co-registered physical fields |
| Observations | `data/observations/sample_NNNN_observations25.npz` | Raw and processed 25-shot gathers |
| Training pair | `data/training_pairs/sample_NNNN_training_pair25.npz` | Quantitative RTM, targets, masks and coordinates |
| RTM quality report | `quality/rtm_reports/sample_NNNN_rtm_report.json` | Artifact metrics, thresholds and admission metadata |

## Geological-property archive

| Keys | Shape and dtype | Unit / axis role |
|---|---|---|
| `x_m`, `z_m`, `seafloor_m`, `bsr_m` | 1229; 1001; 1229; 1229, `float32` | metres; coordinates and horizons |
| `phi_full`, `Sh_full`, `Sg_full`, `Vp_full`, `rho_full` | 1001 × 1229, `float32` | fractions; fractions; fractions; m s⁻¹; kg m⁻³ |
| `phi_view`, `Sh_view`, `Sg_view`, `Vp_view`, `rho_view`, `AI_view` | 512 × 96, `float32` | co-registered vertical × horizontal fields |
| `valid_mask`, `loss_mask` | 512 × 96, `uint8` | support and evaluation masks; neither is an input channel |
| `metadata_json` | scalar Unicode | per-realization generation metadata |

## Observation archive

| Keys | Shape and dtype | Unit / axis role |
|---|---|---|
| `raw_gathers`, `processed_gathers` | 25 × 1501 × 128, `float32` | shot × time × receiver; solver pressure / relative amplitude |
| `runtime_s`, `peak_cuda_memory_gb` | scalar `float64` | seconds; GiB |

## Training-pair archive

| Keys | Shape and dtype | Unit / benchmark role |
|---|---|---|
| `input_rtm_conditioned_unscaled` | 512 × 96, `float32` | quantitative adjoint-gradient input for Tasks 1 and 2 |
| `input_rtm_display_only` | 512 × 96, `float32` | visualization only; never a quantitative input |
| `input_display_scale` | scalar `float64` | per-image display metadata |
| `auxiliary_Vp`, `migration_Vp` | 512 × 96, `float32` | m s⁻¹; Task 1 target source and second input |
| `target_Sh`, `target_Sg` | 512 × 96, `float32` | dimensionless pore-volume fractions; Task 2 targets |
| `valid_mask`, `loss_mask` | 512 × 96, `uint8` | support and loss/evaluation masks; not inputs |
| `raw_representative_gather`, `processed_representative_gather` | 1501 × 128, `float32` | time × receiver visualization gathers |
| `shot_x_m`, `receiver_x_m` | 25; 25 × 128, `float32` | source positions; shot × receiver positions, metres |

Forty-nine released training-pair archives also contain label-aware true-model RTM diagnostics: `diagnostic_true_model_rtm_conditioned_unscaled`, `diagnostic_true_model_rtm_display` and `diagnostic_display_scale`. Reference loaders ignore these optional fields; they are not inputs and are not used for normalization, calibration, loss, metrics or checkpoint selection.

## RTM quality report

The JSON report contains `smooth_rtm_metrics`, `rtm_quality_acceptance`, `rtm_quality_checks`, `admission` and `status`. These record RTM artifact metrics, frozen limits, Boolean checks and release-admission provenance.

## Masks, padding and units

- `valid_mask` marks spatial support.
- `loss_mask` selects the optimization/evaluation region.
- Neither mask is concatenated to a model input.
- Co-registered fields are 512 × 96. The final lateral column is zero padding, not a physical trace.
- Archived `Sh` and `Sg` are fractions of total pore volume, not percentages. Multiply by 100 only for display.
- Dice thresholds are `Sh >= 0.01` (1%) and `Sg >= 0.003` (0.3%).

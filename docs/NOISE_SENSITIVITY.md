# Optional noise sensitivity

```bash
python scripts/add_gather_noise.py \
  --input /path/to/sample_0017_observations25.npz \
  --output outputs/noise/sample_0017_observations25_snr20.npz \
  --key raw_gathers --snr-db 20 --seed 20260824 \
  --reference-time-window 3 6
```

The seeded independent Gaussian noise has the requested per-shot signal-to-noise RMS ratio over the specified time interval (end exclusive). Without a window the entire gather defines the signal RMS; direct arrivals can dominate that value. All NPZ arrays are copied, but only the selected gather array is modified. Noise provenance is stored in `noise_augmentation_json`.

The output is a sensitivity copy. Stored processed gathers and migration images have not been recomputed. If raw gathers are modified, repeat preprocessing and migration before benchmarking noisy image inputs. Do not treat the unchanged stored images as newly computed noisy images; the true property targets can remain unchanged.

This utility does not simulate correlated acquisition noise, ghosts, attenuation, dispersion, elastic effects or 3D propagation. No noise-robustness results are claimed for the original reference models. The five structural-OOD folds concern geological families only.

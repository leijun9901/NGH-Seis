# Geological surfaces and image events

The five selected controls (candidates 0001–0005, one per family) repeat migration with true Vp and unchanged constant density of 1,800 kg/m³. They isolate velocity from density. These are selected examples, not release-wide accuracy estimates.

## Definitions

- **Geometric BSR:** archived `bsr_m - seafloor_m`. This prescribed reference need not be a material discontinuity or an observable BSR. Hydrate/gas coexistence and leakage prevent equating it with every reservoir edge.
- **Gas top:** shallowest linearly interpolated `Sg = 0.003` crossing. It is an operational saturation contour, not automatically an acoustic interface. Missing crossings remain missing; this does not imply strictly zero gas. The script also records support and threshold sensitivity.
- **Deep surfaces:** two constant depositional-coordinate surfaces selected from separated peaks of the median absolute log-impedance derivative at 750–1,120 m below seafloor. Selection uses geology before image inspection. These are stratigraphic markers in a continuous stochastic model, not named sharp lithological boundaries. Generator reconstruction must match frozen Vp, density, seafloor, BSR and lithology exactly.

## Association and limitations

The measured quantity is the depth of an image-envelope energy peak relative to a geological surface. No correlation-window shift is used as a boundary pick. Gates require a 30 m log-impedance contrast of at least 0.015, a distinct true-Vp event within one quarter wavelength, and a dominant released-image peak in a symmetric ±180 m search. A 1D vertical-time estimate is used only to reject inconsistent candidates (quarter wavelength plus 20 m buffer); it is never substituted for a measured position. At least three neighbouring candidates and lateral residual consistency are required. All failures and intermediate values remain in CSV.

These operational gates do not prove unique event identity. Accepted points are **provisional interface–event associations**; offsets include finite bandwidth, gradient point-spread response and constant-density mismatch. Quarter wavelength and 3 m sampling limit interpretation. Trace percentiles describe a spatially correlated selected subset, not confidence intervals. Changing dominance from 1.5 to 1.25/2.0 and buffer from 20 m to 0/40 m is documented separately; this is association-rule sensitivity, not additional migration experiments.

| Family | Geometric BSR | Gas top | Deep surface 1 | Deep surface 2 |
|---|---:|---:|---:|---:|
| Diffuse | unresolved | unresolved | unresolved | unresolved |
| Lenses | unresolved | unresolved | unresolved | unresolved |
| Patchy | +20.9 m (12/95) | +25.2 m (9/24) | +81.0 m (74/95) | +108.8 m (66/95) |
| Chimney | +27.8 m (47/95) | +31.1 m (48/95) | unresolved | unresolved |
| Anticline | unresolved | unresolved | unresolved | +57.2 m (82/95) |

Values are median released-event offsets below defined surfaces; denominators count traces with that surface, not all 1,000 realizations. For patchy gas top, 71/95 traces have no crossing at the declared threshold. Weak contrast, competing events, kinematic inconsistency and inadequate lateral support cause unresolved ties. An unresolved tie does not establish absence of a physical reflector.

## Reproduce the diagnostic

Download and SHA-256 verify the logical archive from the [versioned release](https://github.com/leijun9901/NGH-Seis/releases/tag/v1.0.0), then extract it and run the diagnostic. Small transport parts are joined automatically; both part and complete-file hashes are checked:

```bash
python scripts/fetch_release_asset.py --manifest metadata/interface_asset.json \
  --output outputs/NGH-Seis_v1.0_interface_validation.zip
```

```bash
python scripts/validate_imaging_horizons.py \
  --release /path/to/extracted/release \
  --controls /path/to/extracted/controls \
  --output outputs/interface_validation --plot
```

The package includes byte-identical geology/pair source files, five derived migration controls, a file manifest, reference CSV/JSON and vector figures. Use the public default geology configuration. Generating new migration controls requires the acquisition/processing pipeline and a suitable GPU; this command recomputes geology, associations and plots from the supplied controls.

To regenerate all controls from the complete extracted dataset, use `python scripts/run_imaging_controls.py --release /path/to/NGH-Seis_v1.0 --output outputs/imaging_controls`. This additionally repeats two smooth-model checks and the declared direct-wave-mute sensitivity test; outputs must remain outside the frozen release.

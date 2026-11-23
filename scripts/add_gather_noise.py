"""Create a separate NPZ copy with seeded Gaussian gather noise.

This is an acoustic sensitivity utility, not a model of field acquisition noise.
No published dataset file is overwritten. Reprocess and migrate noisy raw gathers
before using them in image-based benchmarks.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def add_noise(gather, snr_db, seed, time_slice):
    if gather.ndim != 3 or not np.issubdtype(gather.dtype, np.floating) or not np.isfinite(gather).all():
        raise ValueError('Expected finite shot-time-receiver data')
    signal = gather.astype(np.float64)
    noise = np.random.default_rng(seed).standard_normal(signal.shape)
    reference = signal[:, time_slice, :]
    rms_signal = np.sqrt(np.mean(reference**2, axis=(1,2), keepdims=True))
    rms_noise = np.sqrt(np.mean(noise[:, time_slice, :]**2, axis=(1,2), keepdims=True))
    if np.any(rms_signal == 0):
        raise ValueError('Reference time window has zero signal RMS in a shot')
    noise *= rms_signal / (rms_noise * 10**(snr_db/20))
    return (signal + noise).astype(gather.dtype)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--key',choices=['raw_gathers','processed_gathers'],default='raw_gathers')
    p.add_argument('--snr-db',type=float,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--dt-s',type=float,default=.004)
    p.add_argument('--reference-time-window',type=float,nargs=2,metavar=('START','END'))
    args=p.parse_args()
    if args.output.suffix.lower() != '.npz':
        p.error('Output filename must end in .npz')
    if args.output.resolve()==args.input.resolve() or args.output.exists():
        p.error('Output must be a new file outside the original archive')
    if not np.isfinite(args.snr_db) or args.dt_s<=0:
        p.error('SNR must be finite and dt must be positive')
    with np.load(args.input,allow_pickle=False) as f:
        arrays={k:f[k] for k in f.files}
    n=arrays[args.key].shape[1]
    a,b=0,n
    if args.reference_time_window:
        start,end=args.reference_time_window
        a,b=int(round(start/args.dt_s)),int(round(end/args.dt_s))
        if not 0<=a<b<=n:
            p.error('Reference time window is outside the gather')
    arrays[args.key]=add_noise(arrays[args.key],args.snr_db,args.seed,slice(a,b))
    arrays['noise_augmentation_json']=np.array(json.dumps(dict(
        distribution='Gaussian independent samples',snr_db=args.snr_db,seed=args.seed,
        key=args.key,reference_samples=[a,b],dt_s=args.dt_s,
        convention='Per-shot signal/noise RMS ratio in the specified time window',
        derived_arrays='Other arrays are unchanged and have NOT been recomputed from noisy data')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**arrays)


if __name__=='__main__':
    main()

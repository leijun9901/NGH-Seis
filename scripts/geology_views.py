from __future__ import annotations

import argparse

import json

from pathlib import Path

import sys

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

from src.datasets.field_aligned_view import extract_seafloor_relative_view

from src.geology.blake_ew0008 import generate_blake_ew0008

def _extract(field: np.ndarray, seafloor: np.ndarray, config: dict):
    grid = config["grid"]
    crop = config["network_crop"]
    return extract_seafloor_relative_view(
        field,
        seafloor,
        source_dx_m=float(grid["dx_m"]),
        source_dz_m=float(grid["dz_m"]),
        x_start_m=float(crop["x_start_m"]),
        trace_count=int(crop["physical_trace_count"]),
        padded_trace_count=int(crop["padded_trace_count"]),
        output_dx_m=float(crop["output_dx_m"]),
        output_dz_m=float(crop["output_dz_m"]),
        output_nz=int(crop["nz"]),
        water_samples=int(crop["samples_above_seafloor"]),
        interpolation_order=1,
        antialias_lateral=True,
    )

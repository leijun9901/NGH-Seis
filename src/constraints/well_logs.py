"""Validated ingestion and joint block sampling for depth-indexed well logs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = ("well_id", "depth_mbsf", "zone", "phi", "Sh", "Sg", "Vp_mps", "rho_kgm3")
ALLOWED_ZONES = {"hydrate", "mixed", "free_gas", "background"}


def load_well_log_csv(path: str | Path) -> list[dict]:
    """Load real, depth-indexed rows without inventing missing measurements."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required well-log columns: {missing}")
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            row = {"well_id": raw["well_id"].strip(), "zone": raw["zone"].strip()}
            if not row["well_id"] or row["zone"] not in ALLOWED_ZONES:
                raise ValueError(f"Invalid well_id or zone at CSV line {line_number}")
            for name in REQUIRED_COLUMNS[1:]:
                if name == "zone":
                    continue
                text = raw[name].strip()
                if text == "":
                    raise ValueError(
                        f"Missing {name} at CSV line {line_number}; impute explicitly before ingestion."
                    )
                row[name] = float(text)
            if not (0.0 <= row["phi"] <= 0.8):
                raise ValueError(f"Unphysical phi at CSV line {line_number}")
            if min(row["Sh"], row["Sg"]) < 0.0 or row["Sh"] + row["Sg"] > 1.0 + 1e-8:
                raise ValueError(f"Invalid phase saturations at CSV line {line_number}")
            if not (1000.0 <= row["Vp_mps"] <= 6000.0 and 900.0 <= row["rho_kgm3"] <= 3500.0):
                raise ValueError(f"Implausible Vp or density at CSV line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("Well-log CSV contains no observations")
    return rows


def moving_block_sample(
    rows: list[dict],
    *,
    block_length: int,
    seed: int,
    blocks_per_group: int = 1,
) -> list[list[dict]]:
    """Sample contiguous joint-property blocks within each well and zone.

    Each returned row keeps phi, Sh, Sg, Vp, and density together.  This avoids
    destroying measured cross-property and short-range depth relationships.
    """
    if block_length < 2 or blocks_per_group < 1:
        raise ValueError("block_length must be >=2 and blocks_per_group >=1")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["well_id"], row["zone"])].append(row)
    rng = np.random.default_rng(seed)
    output = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: row["depth_mbsf"])
        if len(group) < block_length:
            continue
        for _ in range(blocks_per_group):
            start = int(rng.integers(0, len(group) - block_length + 1))
            output.append([dict(row) for row in group[start : start + block_length]])
    if not output:
        raise ValueError("No well/zone group is long enough for the requested block")
    return output

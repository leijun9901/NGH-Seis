"""Balanced parent-geology design for NGH-Seis v1.0.

The module separates three sources of variation:
1. well-family constraints,
2. parent 2-D geology,
3. conditional saturation/rock-physics realizations.

It does not claim to resample raw well logs unless an explicit depth-indexed
CSV is provided.  The default configuration is a literature-summary proxy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.geology.diverse_shenhu import generate_diverse_shenhu
from src.rock_physics import shenhu_log_calibrated_properties


def _stable_uniform(seed: int, count: int, dimensions: int) -> np.ndarray:
    """Return a maximin-like Latin hypercube without extra dependencies."""
    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    best_score = -np.inf
    for _ in range(32):
        candidate = np.empty((count, dimensions), dtype=np.float64)
        for dim in range(dimensions):
            candidate[:, dim] = (rng.permutation(count) + rng.random(count)) / count
        if count == 1:
            return candidate
        delta = candidate[:, None, :] - candidate[None, :, :]
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        distance[np.diag_indices_from(distance)] = np.inf
        score = float(distance.min())
        if score > best_score:
            best, best_score = candidate, score
    assert best is not None
    return best


def _parent_hash(family: str, style: str, parent_index: int) -> str:
    text = f"{family}|{style}|{parent_index}".encode("utf-8")
    return hashlib.sha1(text).hexdigest()[:12]


def build_hierarchical_design(config_path: str | Path, seed: int = 20260813) -> list[dict]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    design = config["design"]
    families = design["well_families"]
    styles = design["structural_styles"]
    parent_spec = design["parent_geologies_per_cell"]
    nreal = int(design["realizations_per_parent"])
    records: list[dict] = []
    cell_index = 0
    for family in families:
        for style in styles:
            nparents = (
                int(parent_spec[f"{family}|{style}"])
                if isinstance(parent_spec, dict)
                else int(parent_spec)
            )
            lhs = _stable_uniform(seed + cell_index * 1009, nparents, 8)
            for parent_index, vector in enumerate(lhs):
                parent_id = _parent_hash(family, style, parent_index)
                parent_seed = seed + cell_index * 100_003 + parent_index * 101
                parent_parameters = {
                    "water_depth_q": float(vector[0]),
                    "hydrate_top_q": float(vector[1]),
                    "regional_dip_q": float(vector[2]),
                    "relief_amplitude_q": float(vector[3]),
                    "lateral_scale_q": float(vector[4]),
                    "thickness_q": float(vector[5]),
                    "style_parameter_1_q": float(vector[6]),
                    "style_parameter_2_q": float(vector[7]),
                }
                for realization_index in range(nreal):
                    realization_seed = parent_seed + (realization_index + 1) * 1_000_003
                    records.append({
                        "sample_index": len(records) + 1,
                        "well_family": family,
                        "structural_style": style,
                        "parent_geology_id": parent_id,
                        "parent_index_in_cell": parent_index,
                        "realization_index": realization_index,
                        "parent_seed": parent_seed,
                        "realization_seed": realization_seed,
                        "constraint_mode": config["constraint_mode"],
                        "parent_parameters": parent_parameters,
                    })
            cell_index += 1
    expected = int(design["formal_sample_count"])
    if len(records) != expected:
        raise ValueError(f"Design produced {len(records)} samples; expected {expected}.")
    return records


def split_by_parent(records: list[dict], seed: int = 20260813) -> dict[str, list[dict]]:
    """Create a leakage-safe 70/15/15 split at parent-geology level."""
    parent_ids = sorted({record["parent_geology_id"] for record in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(parent_ids)
    n = len(parent_ids)
    ntrain, nval = round(0.70 * n), round(0.15 * n)
    mapping = {
        parent_id: "train" if i < ntrain else "val" if i < ntrain + nval else "test"
        for i, parent_id in enumerate(parent_ids)
    }
    output = {"train": [], "val": [], "test": []}
    for record in records:
        output[mapping[record["parent_geology_id"]]].append(record)
    return output


def write_design(config_path: str | Path, output_path: str | Path) -> None:
    records = build_hierarchical_design(config_path)
    splits = split_by_parent(records)
    payload = {
        "records": records,
        "split_counts": {name: len(items) for name, items in splits.items()},
        "split_parent_counts": {
            name: len({item["parent_geology_id"] for item in items})
            for name, items in splits.items()
        },
        "split_assignment": {
            item["parent_geology_id"]: name for name, items in splits.items() for item in items
        },
    }
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_from_design_record(
    generator_config_path: str | Path,
    record: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    """Materialize one geological realization from a design-table record."""
    arrays, metadata = generate_diverse_shenhu(
        generator_config_path,
        sample_index=int(record["sample_index"]),
        seed=int(record["realization_seed"]),
        family_name_override=record["well_family"],
        style_override=record["structural_style"],
        geometry_seed=int(record["parent_seed"]),
        property_seed=int(record["realization_seed"]),
        parent_parameters=record["parent_parameters"],
    )
    metadata.update({
        "well_family_design": record["well_family"],
        "parent_geology_id": record["parent_geology_id"],
        "realization_index": record["realization_index"],
        "constraint_mode": record["constraint_mode"],
        "parent_parameters": record["parent_parameters"],
    })
    return arrays, metadata


def generate_velocity_valid_from_record(
    generator_config_path: str | Path,
    record: dict,
    *,
    maximum_attempts: int | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Resample properties, but never parent geometry, until Vp checks pass."""
    config = json.loads(Path(generator_config_path).read_text(encoding="utf-8"))
    attempts = maximum_attempts or int(config.get("maximum_realization_attempts", 40))
    base_seed = int(record["realization_seed"])
    rejected_diagnostics: list[dict] = []
    for attempt in range(attempts):
        candidate = dict(record)
        # Derive an attempt seed from immutable record identifiers.  A simple
        # arithmetic increment can collide with the base seed of the next
        # realization and silently create duplicate conditional models.
        seed_text = (
            f'{record["parent_geology_id"]}|{record["realization_index"]}|'
            f'{base_seed}|attempt={attempt}'
        ).encode("utf-8")
        candidate["realization_seed"] = int.from_bytes(
            hashlib.sha256(seed_text).digest()[:8], "little"
        ) % (2**63 - 1)
        geo, metadata = generate_from_design_record(generator_config_path, candidate)
        grid = config["grid"]
        water = (
            np.arange(grid["nz"])[:, None] * grid["dz_m"]
            < geo["seafloor_m"][None, :]
        )
        vp, rho = shenhu_log_calibrated_properties(
            geo["phi"], geo["Sh"], geo["Sg"], water,
            hydrate_response_scale=metadata["hydrate_response_scale"],
            gas_response_scale=metadata["gas_response_scale"],
        )
        zone_means = {
            name: float(vp[geo[f"zone_{name}"].astype(bool)].mean())
            for name in ("hydrate", "mixed", "free_gas")
        }
        global_limits = config["velocity_acceptance_m_per_s"]
        zone_limits = config["zone_mean_velocity_acceptance_m_per_s"]
        valid = (
            float(vp.min()) >= global_limits["minimum"]
            and float(vp.max()) <= global_limits["maximum"]
            and all(zone_limits[name][0] <= value <= zone_limits[name][1]
                    for name, value in zone_means.items())
        )
        if valid:
            geo = {**geo, "vp": vp, "rho": rho}
            metadata.update({
                "requested_realization_seed": base_seed,
                "accepted_realization_seed": candidate["realization_seed"],
                "acceptance_attempts": attempt + 1,
                "zone_mean_vp_mps": zone_means,
            })
            return geo, metadata
        rejected_diagnostics.append({
            "vp_min": float(vp.min()),
            "vp_max": float(vp.max()),
            **{f"{name}_mean": value for name, value in zone_means.items()},
        })
    diagnostic_ranges = {
        key: [min(item[key] for item in rejected_diagnostics),
              max(item[key] for item in rejected_diagnostics)]
        for key in rejected_diagnostics[0]
    }
    raise ValueError(
        f'Sample {record["sample_index"]} failed velocity acceptance after {attempts} attempts; '
        f'observed ranges={diagnostic_ranges}'
    )

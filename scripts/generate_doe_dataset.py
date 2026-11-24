"""Generate real multi-topology ANSYS MAPDL DOE dataset for AeroRF-Heal Phase 2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import List

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerorf.ansys_runner import (
    DAMAGE_CLASSES,
    DOELoadSample,
    run_doe_topology_batch,
)
from aerorf.config import ArrayConfig, FBGSensorConfig, WingboxConfig


def sample_lhs_cases(
    per_class: int,
    damage_zone: str,
    seed: int,
) -> List[DOELoadSample]:
    """Generate space-filling Latin Hypercube samples across flight, thermal & damage space."""
    # Bounds: p0, torque_grad, p_gust, x_gust, dt_upper, dt_lower, dt_radar, damage_factor
    l_bounds = np.array([-15000.0, -0.80, -15000.0, 0.25, -25.0, -25.0, 0.0, 0.10])
    u_bounds = np.array([48000.0, 0.90, 20000.0, 1.05, 65.0, 20.0, 45.0, 0.65])

    sampler = qmc.LatinHypercube(d=8, seed=seed)
    unit = sampler.random(n=per_class)
    scaled = qmc.scale(unit, l_bounds, u_bounds)

    samples: List[DOELoadSample] = []
    for row in scaled:
        df = 1.0 if damage_zone == "intact" else float(row[7])
        samples.append(
            DOELoadSample(
                p0_pa=float(row[0]),
                torque_grad=float(row[1]),
                p_gust_pa=float(row[2]),
                x_gust_m=float(row[3]),
                dt_upper_c=float(row[4]),
                dt_lower_c=float(row[5]),
                dt_radar_c=float(row[6]),
                damage_factor=df,
            )
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AeroRF-Heal ANSYS DOE dataset.")
    parser.add_argument("--per-class", type=int, default=30, help="Cases per damage topology class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for LHS sampling.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "aerorf_doe_dataset.npz",
        help="Output .npz dataset path.",
    )
    args = parser.parse_args()

    wing_cfg = WingboxConfig()
    array_cfg = ArrayConfig(z_surface=wing_cfg.z_top)
    fbg_cfg = FBGSensorConfig()

    doe_root = ROOT / "outputs" / "doe_batches"
    doe_root.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_fbg = []
    all_disp = []
    all_rot = []
    all_labels = []
    all_df = []
    all_params = []

    t_start = time.perf_counter()
    print(
        f"Generating {args.per_class * len(DAMAGE_CLASSES)} real ANSYS MAPDL cases "
        f"({args.per_class} x {len(DAMAGE_CLASSES)} classes)..."
    )

    for cls_idx, damage_zone in enumerate(DAMAGE_CLASSES):
        samples = sample_lhs_cases(args.per_class, damage_zone, seed=args.seed + cls_idx * 101)
        batch_dir = doe_root / damage_zone
        print(f"\n[{cls_idx + 1}/{len(DAMAGE_CLASSES)}] Running MAPDL DOE batch: {damage_zone} ({len(samples)} cases)...")
        fbg_s, arr_d, arr_r, run_res = run_doe_topology_batch(
            work_dir=batch_dir,
            wing_cfg=wing_cfg,
            array_cfg=array_cfg,
            fbg_cfg=fbg_cfg,
            damage_zone=damage_zone,
            samples=samples,
            nproc=4,
            timeout=900,
        )
        print(
            f"  Completed in {run_res.elapsed_sec:.2f}s | Nodes: {run_res.node_count} | "
            f"Errors: {len(run_res.errors)} | Mean |UZ|max: {np.abs(arr_d[:, :, 2]).max(axis=1).mean()*1e3:.2f} mm"
        )
        assert np.all(np.isfinite(fbg_s)) and np.all(np.isfinite(arr_d)) and np.all(np.isfinite(arr_r))

        all_fbg.append(fbg_s)
        all_disp.append(arr_d)
        all_rot.append(arr_r)
        all_labels.append(np.full(len(samples), cls_idx, dtype=np.int64))
        all_df.append(np.array([s.damage_factor for s in samples], dtype=np.float64))
        all_params.append(
            np.array(
                [
                    [
                        s.p0_pa,
                        s.torque_grad,
                        s.p_gust_pa,
                        s.x_gust_m,
                        s.dt_upper_c,
                        s.dt_lower_c,
                        s.dt_radar_c,
                        s.damage_factor,
                    ]
                    for s in samples
                ],
                dtype=np.float64,
            )
        )

    fbg_strains = np.concatenate(all_fbg, axis=0)
    array_disp = np.concatenate(all_disp, axis=0)
    array_rot = np.concatenate(all_rot, axis=0)
    damage_label = np.concatenate(all_labels, axis=0)
    damage_factor = np.concatenate(all_df, axis=0)
    load_params = np.concatenate(all_params, axis=0)

    np.savez_compressed(
        args.out,
        fbg_strains=fbg_strains,
        array_disp=array_disp,
        array_rot=array_rot,
        damage_label=damage_label,
        damage_factor=damage_factor,
        load_params=load_params,
        class_names=np.array(DAMAGE_CLASSES),
    )

    total_sec = time.perf_counter() - t_start
    print("\n=== DOE DATASET GENERATION SUMMARY ===")
    print(f"Saved dataset    : {args.out}")
    print(f"Total cases      : {len(damage_label)} ({total_sec:.1f}s total, {total_sec/len(damage_label):.2f}s/case)")
    print(f"fbg_strains shape: {fbg_strains.shape} | peak |strain|: {np.abs(fbg_strains).max()*1e6:.1f} ue")
    print(f"array_disp shape : {array_disp.shape} | peak |UZ|: {np.abs(array_disp[:, :, 2]).max()*1e3:.2f} mm")


if __name__ == "__main__":
    main()

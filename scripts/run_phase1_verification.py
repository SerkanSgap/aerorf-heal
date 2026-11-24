"""Phase 1 Verification & Coupled PoC: ANSYS MAPDL 2026 R1 + X-Band 16x16 Self-Healing."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerorf.ansys_runner import (
    find_mapdl_executable,
    generate_cantilever_vv_apdl,
    generate_wingbox_apdl,
    interpolate_to_array,
    load_fbg_sensors,
    load_upper_skin_nodes,
    run_mapdl_batch,
)
from aerorf.config import ArrayConfig, FBGSensorConfig, WingLoadCase, WingboxConfig
from aerorf.rf_array import evaluate_conformal_scenario


def run_cantilever_vv(out_root: Path) -> dict:
    """Run ANSYS Cantilever Plate V&V benchmark vs analytical Euler-Bernoulli theory."""
    vv_dir = out_root / "vv_cantilever"
    vv_dir.mkdir(parents=True, exist_ok=True)
    apdl_path = vv_dir / "cantilever_vv.mac"
    csv_name = "cantilever_vv.csv"

    generate_cantilever_vv_apdl(apdl_path, output_csv_name=csv_name)
    res = run_mapdl_batch(apdl_path, vv_dir, nproc=4, timeout=180, jobname="vv_cant")
    if not res.ok:
        raise RuntimeError(f"Cantilever V&V MAPDL failed: {res.errors}")

    csv_path = vv_dir / csv_name
    row = pd.read_csv(csv_path, header=None).iloc[0]
    node_count = int(row[0])
    elem_count = int(row[1])
    uz_ansys = float(row[2])

    # Analytical: L=1.0, b=0.05, t=0.01, E=70e9, F=10.0
    i_yy = 0.05 * (0.01**3) / 12.0
    uz_analytic = 10.0 * (1.0**3) / (3.0 * 70.0e9 * i_yy)
    rel_err_pct = 100.0 * abs(uz_ansys - uz_analytic) / uz_analytic

    print("\n=== [1] ANSYS MAPDL CANTILEVER PLATE V&V ===")
    print(f"Nodes / Elements : {node_count} / {elem_count} (Solve: {res.elapsed_sec:.2f}s)")
    print(f"Analytical Tip UZ: {uz_analytic * 1e3:.4f} mm")
    print(f"ANSYS SHELL181 UZ: {uz_ansys * 1e3:.4f} mm")
    print(f"Relative Error   : {rel_err_pct:.3f}% (Target < 2.5%)")
    assert rel_err_pct < 2.5, f"V&V error too high: {rel_err_pct:.3f}%"

    return {
        "nodes": node_count,
        "elements": elem_count,
        "elapsed_sec": res.elapsed_sec,
        "uz_analytic_mm": uz_analytic * 1e3,
        "uz_ansys_mm": uz_ansys * 1e3,
        "rel_err_pct": rel_err_pct,
    }


def build_poc_html(
    html_path: Path,
    skin_df: pd.DataFrame,
    fbg_df: pd.DataFrame,
    rf_res: dict,
    load_case: WingLoadCase,
) -> Path:
    """Create interactive 2-panel Plotly PoC showcase (3D deformed wing + RF healing)."""
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.52, 0.48],
        subplot_titles=(
            "3D Deformed Composite Wing Skin (ANSYS MAPDL) + 256 AESA Elements + 12 FBGs",
            f"X-Band 10 GHz Pattern Cut (Scan θ₀ = +{rf_res['theta0_deg']:.0f}°, φ₀ = {rf_res['phi0_deg']:.0f}°)",
        ),
        horizontal_spacing=0.06,
    )

    # Panel 1: Triangulated upper skin surface colored by UZ (mm)
    xy = skin_df[["x", "y"]].to_numpy()
    tri = Delaunay(xy)
    exag = 2.0  # Subtle 2x visual displacement scale for clear 3D inspection
    x_def = skin_df["x"].to_numpy() + exag * skin_df["ux"].to_numpy()
    y_def = skin_df["y"].to_numpy() + exag * skin_df["uy"].to_numpy()
    z_def = skin_df["z"].to_numpy() + exag * skin_df["uz"].to_numpy()
    uz_mm = skin_df["uz"].to_numpy() * 1e3

    fig.add_trace(
        go.Mesh3d(
            x=x_def,
            y=y_def,
            z=z_def,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            intensity=uz_mm,
            colorscale="Turbo",
            colorbar=dict(title="UZ (mm)", x=0.47, len=0.75),
            opacity=0.88,
            name="Upper Composite Skin (2x def)",
            hovertemplate="x=%{x:.3f}m<br>y=%{y:.3f}m<br>UZ=%{intensity:.2f}mm<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Deformed AESA element positions (same 2x visual exaggeration relative to z_surface)
    pos_nom = rf_res["positions_nominal"]
    pos_def = rf_res["positions_deformed"]
    disp = pos_def - pos_nom
    pos_vis = pos_nom + exag * disp

    fig.add_trace(
        go.Scatter3d(
            x=pos_vis[:, 0],
            y=pos_vis[:, 1],
            z=pos_vis[:, 2] + 0.004,
            mode="markers",
            marker=dict(size=3.5, color="#FFD166", symbol="square"),
            name="16×16 X-Band AESA (256 elem)",
        ),
        row=1,
        col=1,
    )

    # Place FBG visual markers on nearest deformed skin node
    skin_xy = skin_df[["x", "y"]].to_numpy()
    fbg_vis_coords = []
    for _, r in fbg_df.iterrows():
        d2 = (skin_xy[:, 0] - r["x_node"]) ** 2 + (skin_xy[:, 1] - r["y_node"]) ** 2
        idx = int(np.argmin(d2))
        fbg_vis_coords.append((x_def[idx], y_def[idx], z_def[idx] + 0.006, r["ueps_x"]))
    fbg_arr = np.asarray(fbg_vis_coords)

    fig.add_trace(
        go.Scatter3d(
            x=fbg_arr[:, 0],
            y=fbg_arr[:, 1],
            z=fbg_arr[:, 2],
            mode="markers",
            marker=dict(size=6.5, color="#00F5D4", symbol="diamond", line=dict(width=1, color="black")),
            text=[f"FBG-{i+1}: {eps:.0f} με" for i, eps in enumerate(fbg_arr[:, 3])],
            hoverinfo="text",
            name="12 FBG Strain Sensors",
        ),
        row=1,
        col=1,
    )

    # Panel 2: Radiation pattern comparison
    theta = rf_res["theta_deg"]
    cuts = rf_res["cuts_primary"]
    m_nom = rf_res["metrics_primary"]["nominal"]
    m_unc = rf_res["metrics_primary"]["deformed_uncompensated"]
    m_heal = rf_res["metrics_primary"]["healed_quantized"]

    fig.add_trace(
        go.Scatter(
            x=theta,
            y=cuts["nominal"],
            mode="lines",
            line=dict(color="#A8B2C1", width=2, dash="dash"),
            name=f"Nominal (SLL={m_nom['sll_db']:.1f} dB)",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=theta,
            y=cuts["deformed_uncompensated"],
            mode="lines",
            line=dict(color="#FF4D6D", width=2.5),
            name=f"5G+Thermal Uncomp (Loss={m_unc['gain_loss_db']:.2f} dB, Err={m_unc['boresight_error_deg']:+.2f}°)",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=theta,
            y=cuts["healed_quantized"],
            mode="lines",
            line=dict(color="#06D6A0", width=2.8),
            name=f"6-Bit Self-Healed (Loss={m_heal['gain_loss_db']:.2f} dB, SLL={m_heal['sll_db']:.1f} dB)",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[rf_res["theta0_deg"], rf_res["theta0_deg"]],
            y=[-50.0, 3.0],
            mode="lines",
            line=dict(color="#FFD166", width=1.8, dash="dot"),
            name=f"Target +{rf_res['theta0_deg']:.0f}°",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=(
                f"AeroRF-Heal Phase 1 Coupled Proof-of-Concept — {load_case.name} "
                f"(p0={load_case.p0_pa/1e3:.0f} kPa, ΔT_top=+{load_case.delta_t_upper_c:.0f}°C)"
            ),
            x=0.5,
            font=dict(size=18),
        ),
        height=720,
        width=1480,
        legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="center", x=0.5),
    )
    fig.update_scenes(
        xaxis_title="Span X (m)",
        yaxis_title="Chord Y (m)",
        zaxis_title="Z (2x def, m)",
        aspectmode="data",
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Scan Angle θ (deg)", range=[-60, 60], row=1, col=2)
    fig.update_yaxes(title_text="Normalized Gain (dB)", range=[-50, 3], row=1, col=2)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    return html_path


def main() -> None:
    out_root = ROOT / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    exe = find_mapdl_executable()
    print(f"Detected ANSYS MAPDL executable: {exe}")

    # 1. ANSYS Cantilever V&V
    run_cantilever_vv(out_root)

    # 2. Composite Wingbox Coupled 5G Maneuver + Thermal Run
    wing_cfg = WingboxConfig()
    array_cfg = ArrayConfig(z_surface=wing_cfg.z_top)
    fbg_cfg = FBGSensorConfig()
    load_case = WingLoadCase(
        name="5G_maneuver_thermal",
        p0_pa=35000.0,
        torque_grad=0.65,
        delta_t_upper_c=45.0,
        delta_t_lower_c=5.0,
    )

    wb_dir = out_root / "wingbox_5g_thermal"
    wb_apdl = wb_dir / "wingbox.mac"
    generate_wingbox_apdl(wb_apdl, wing_cfg, fbg_cfg, load_case)

    print("\n=== [2] ANSYS COMPOSITE UAV WINGBOX (5G + TORSION + +45C THERMAL) ===")
    wb_res = run_mapdl_batch(wb_apdl, wb_dir, nproc=4, timeout=180, jobname="wingbox")
    if not wb_res.ok:
        raise RuntimeError(f"Wingbox MAPDL failed:\n" + "\n".join(wb_res.errors))

    print(
        f"Nodes / Elements : {wb_res.node_count} / {wb_res.element_count} "
        f"(Solve: {wb_res.elapsed_sec:.2f}s, Errors: {len(wb_res.errors)})"
    )
    assert wb_res.node_count < 15000, f"Unexpectedly high node count: {wb_res.node_count}"

    skin_csv = wb_dir / "upper_skin_nodes.csv"
    fbg_csv = wb_dir / "fbg_sensors.csv"
    skin_df = load_upper_skin_nodes(skin_csv)
    fbg_df = load_fbg_sensors(fbg_csv)

    max_uz_mm = skin_df["uz"].max() * 1e3
    min_uz_mm = skin_df["uz"].min() * 1e3
    max_abs_ueps = fbg_df[["ueps_x", "ueps_y"]].abs().to_numpy().max()
    min_abs_ueps = fbg_df["ueps_x"].abs().min()

    print(f"Upper Skin Nodes : {len(skin_df)}")
    print(f"Wing Deflection  : min UZ = {min_uz_mm:.2f} mm, tip max UZ = {max_uz_mm:.2f} mm")
    print(
        f"12 FBG Sensors   : |eps_x| min = {min_abs_ueps:.1f} ue, "
        f"peak strain = {max_abs_ueps:.1f} ue"
    )
    assert len(fbg_df) == 12
    assert np.all(np.isfinite(fbg_df[["eps_x", "eps_y", "eps_xy"]].to_numpy()))
    assert max_abs_ueps > 10.0, "Extracted FBG strains are unexpectedly near zero!"

    # 3. Interpolate onto 256 conformal AESA elements & run RF self-healing
    disp_xyz, rot_xyz = interpolate_to_array(skin_csv, array_cfg)
    assert disp_xyz.shape == (256, 3) and np.all(np.isfinite(disp_xyz))
    assert rot_xyz.shape == (256, 3) and np.all(np.isfinite(rot_xyz))

    ap_dz_mm = (disp_xyz[:, 2].max() - disp_xyz[:, 2].min()) * 1e3
    ap_mean_dz_mm = disp_xyz[:, 2].mean() * 1e3
    print(
        f"Aperture Footprint: mean UZ = {ap_mean_dz_mm:.2f} mm, "
        f"peak-to-valley delta_UZ = {ap_dz_mm:.2f} mm ({ap_dz_mm / (array_cfg.lambda0 * 1e3):.2f} lambda0)"
    )

    rf_res = evaluate_conformal_scenario(
        array_cfg,
        disp_xyz,
        rot_xyz,
        theta0_deg=15.0,
        phi0_deg=0.0,
        quant_bits=6,
    )

    print("\n=== [3] X-BAND (10 GHz) PHASED-ARRAY PERFORMANCE AT θ₀ = +15°, φ₀ = 0° ===")
    print(f"{'Mode':<26} {'Gain Loss (dB)':>14} {'Boresight Err':>15} {'SLL (dB)':>11} {'HPBW (deg)':>12}")
    print("-" * 82)
    for key, label in (
        ("nominal", "Nominal (Undeformed)"),
        ("deformed_uncompensated", "5G+Thermal Uncompensated"),
        ("healed_quantized", "6-Bit Self-Healed"),
        ("healed_ideal", "Ideal Phase Self-Healed"),
    ):
        m = rf_res["metrics_primary"][key]
        print(
            f"{label:<26} {m['gain_loss_db']:>14.3f} "
            f"{m['boresight_error_deg']:>+14.3f}° {m['sll_db']:>11.2f} {m['hpbw_deg']:>12.2f}"
        )

    # 4. Generate interactive Plotly PoC
    poc_html = out_root / "phase1_coupled_poc.html"
    build_poc_html(poc_html, skin_df, fbg_df, rf_res, load_case)
    print(f"\nInteractive PoC generated: {poc_html}")


if __name__ == "__main__":
    main()

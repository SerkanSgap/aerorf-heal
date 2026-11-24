"""Pure compute backend and Plotly figure builders for AeroRF-Heal Streamlit cockpit."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import StratifiedShuffleSplit
import torch

from .ansys_runner import DAMAGE_CLASSES
from .config import ArrayConfig, FBGSensorConfig, WingboxConfig
from .rf_array import compute_pattern_uv_grid, evaluate_conformal_scenario
from .shape_ai import (
    AeroRFShapeNet,
    PODBasis,
    RidgeBaseline,
    load_checkpoint,
    predict_with_latency,
)


@dataclass
class CockpitAssets:
    """Loaded dataset, trained ShapeNet v2, Ridge baseline, and benchmark metadata."""

    project_root: Path
    wing_cfg: WingboxConfig
    array_cfg: ArrayConfig
    fbg_cfg: FBGSensorConfig
    fbg_strains: np.ndarray
    array_disp: np.ndarray
    array_rot: np.ndarray
    damage_label: np.ndarray
    damage_factor: np.ndarray
    load_params: np.ndarray
    class_names: List[str]
    heldout_mask: np.ndarray
    model: AeroRFShapeNet
    pod: PODBasis
    strain_mean: np.ndarray
    strain_std: np.ndarray
    ridge: RidgeBaseline
    benchmark: Dict[str, Any]


def load_cockpit_assets(project_root: Path) -> CockpitAssets:
    """Load real ANSYS DOE dataset, trained ShapeNet checkpoint, and Ridge baseline."""
    project_root = Path(project_root)
    wing_cfg = WingboxConfig()
    array_cfg = ArrayConfig(z_surface=wing_cfg.z_top)
    fbg_cfg = FBGSensorConfig()

    raw = np.load(project_root / "data" / "aerorf_doe_dataset.npz", allow_pickle=False)
    fbg_strains = raw["fbg_strains"]
    array_disp = raw["array_disp"]
    array_rot = raw["array_rot"]
    damage_label = raw["damage_label"]
    damage_factor = raw["damage_factor"]
    load_params = raw["load_params"]
    class_names = [str(c) for c in raw["class_names"]]

    # Exact stratified 80/20 split matching train_and_eval_ai.py (no data leakage)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, te_idx = next(splitter.split(fbg_strains, damage_label))
    heldout_mask = np.zeros(len(damage_label), dtype=bool)
    heldout_mask[te_idx] = True

    ridge = RidgeBaseline().fit(fbg_strains[tr_idx], array_disp[tr_idx])

    model, pod, strain_mean, strain_std = load_checkpoint(
        project_root / "models" / "aerorf_shapenet.pt"
    )

    bench_path = project_root / "outputs" / "phase2_ai_benchmark.json"
    benchmark = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.is_file() else {}

    return CockpitAssets(
        project_root=project_root,
        wing_cfg=wing_cfg,
        array_cfg=array_cfg,
        fbg_cfg=fbg_cfg,
        fbg_strains=fbg_strains,
        array_disp=array_disp,
        array_rot=array_rot,
        damage_label=damage_label,
        damage_factor=damage_factor,
        load_params=load_params,
        class_names=class_names,
        heldout_mask=heldout_mask,
        model=model,
        pod=pod,
        strain_mean=strain_mean,
        strain_std=strain_std,
        ridge=ridge,
        benchmark=benchmark,
    )


def run_live_combat_inference(
    assets: CockpitAssets,
    case_idx: int,
    dead_sensor_ids: Sequence[int],
    noise_pct: float = 3.0,
    theta0_deg: float = 15.0,
    quant_bits: int | None = 6,
    rng_seed: int = 777,
) -> Dict[str, Any]:
    """Run live masked FBG combat inference, parity residual check, and X-band RF healing."""
    rng = np.random.default_rng(rng_seed + int(case_idx))
    raw_fbg = assets.fbg_strains[case_idx : case_idx + 1]
    strains_norm = (raw_fbg - assets.strain_mean) / assets.strain_std

    noise_std = max(0.0, float(noise_pct) / 100.0)
    noisy = strains_norm.copy()
    if noise_std > 0.0:
        noisy += rng.normal(0.0, noise_std, size=noisy.shape)

    mask_12 = np.ones((1, assets.fbg_cfg.num_sensors), dtype=np.float64)
    dead_clean = sorted({int(s) for s in dead_sensor_ids if 0 <= int(s) < assets.fbg_cfg.num_sensors})
    if dead_clean:
        mask_12[0, dead_clean] = 0.0
        noisy[0, dead_clean, :] = 0.0

    s36 = noisy.reshape(1, -1)
    x_48 = np.concatenate([s36, mask_12], axis=1)

    disp_ai_b, _, _, latency_ms = predict_with_latency(
        assets.model, assets.pod, x_48, n_warmup=3, n_bench=25
    )
    disp_ai = disp_ai_b[0]
    disp_ridge = assets.ridge.predict(s36)[0]
    disp_true = assets.array_disp[case_idx]
    rot_true = assets.array_rot[case_idx]

    device = next(assets.model.parameters()).device
    with torch.no_grad():
        xt = torch.tensor(x_48, dtype=torch.float32, device=device)
        s36_t = xt[:, :36]
        m12_t = xt[:, 36:]
        s_imp = assets.model.impute_masked(s36_t, m12_t)
        m36_t = m12_t.repeat_interleave(3, dim=1)
        r_raw = (s_imp - s_imp @ assets.model.w_compat.T) * m36_t
        s_norm = torch.norm(s_imp * m36_t, dim=1, keepdim=True) + 0.1
        r_norm = (r_raw / s_norm).view(12, 3).cpu().numpy()

        _, logits, sev_t = assets.model(xt)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_sev = float(sev_t[0, 0].cpu().item())

    sensor_parity_scores = np.linalg.norm(r_norm, axis=1)
    pred_cls_idx = int(np.argmax(probs))
    true_cls_idx = int(assets.damage_label[case_idx])

    res_ai = evaluate_conformal_scenario(
        assets.array_cfg,
        disp_true,
        rot_true,
        theta0_deg=float(theta0_deg),
        phi0_deg=0.0,
        est_disp_xyz=disp_ai,
        quant_bits=quant_bits,
        n_theta=1801,
    )
    res_ridge = evaluate_conformal_scenario(
        assets.array_cfg,
        disp_true,
        rot_true,
        theta0_deg=float(theta0_deg),
        phi0_deg=0.0,
        est_disp_xyz=disp_ridge,
        quant_bits=quant_bits,
        n_theta=1801,
    )
    res_oracle = evaluate_conformal_scenario(
        assets.array_cfg,
        disp_true,
        rot_true,
        theta0_deg=float(theta0_deg),
        phi0_deg=0.0,
        est_disp_xyz=disp_true,
        quant_bits=quant_bits,
        n_theta=1801,
    )

    # 2D (u, v) beam heatmaps: uncompensated vs AI blind self-healed
    u_vals = np.linspace(-0.85, 0.85, 45)
    v_vals = np.linspace(-0.85, 0.85, 45)
    pos_def = res_ai["positions_deformed"]
    norm_def = res_ai["normals_deformed"]
    ref_peak = float(res_ai["ref_peak_linear"])
    k0 = assets.array_cfg.k0
    q_elem = assets.array_cfg.q_elem

    uv_uncomp = compute_pattern_uv_grid(
        pos_def,
        norm_def,
        res_ai["weights"]["nominal"],
        k0,
        u_vals,
        v_vals,
        q_elem=q_elem,
        ref_peak_linear=ref_peak,
    )
    uv_healed = compute_pattern_uv_grid(
        pos_def,
        norm_def,
        res_ai["weights"]["healed_quantized"],
        k0,
        u_vals,
        v_vals,
        q_elem=q_elem,
        ref_peak_linear=ref_peak,
    )

    ai_rmse_mm = float(np.sqrt(np.mean((disp_ai - disp_true) ** 2)) * 1e3)
    ridge_rmse_mm = float(np.sqrt(np.mean((disp_ridge - disp_true) ** 2)) * 1e3)

    return {
        "case_idx": int(case_idx),
        "is_heldout": bool(assets.heldout_mask[case_idx]),
        "true_class": assets.class_names[true_cls_idx],
        "pred_class": assets.class_names[pred_cls_idx],
        "class_probs": probs,
        "confidence_pct": float(probs[pred_cls_idx] * 100.0),
        "true_severity": float(assets.damage_factor[case_idx]),
        "pred_severity": pred_sev,
        "load_params": assets.load_params[case_idx],
        "dead_sensor_ids": dead_clean,
        "latency_ms": float(latency_ms),
        "ai_rmse_mm": ai_rmse_mm,
        "ridge_rmse_mm": ridge_rmse_mm,
        "disp_true": disp_true,
        "disp_ai": disp_ai,
        "sensor_parity_scores": sensor_parity_scores,
        "raw_fbg_ue": raw_fbg[0] * 1e6,
        "theta_deg": res_ai["theta_deg"],
        "theta0_deg": float(theta0_deg),
        "quant_bits": quant_bits,
        "cuts": {
            "nominal": res_ai["cuts_primary"]["nominal"],
            "uncompensated": res_ai["cuts_primary"]["deformed_uncompensated"],
            "ridge_blind": res_ridge["cuts_primary"]["healed_quantized"],
            "ai_blind": res_ai["cuts_primary"]["healed_quantized"],
        },
        "metrics": {
            "nominal": res_ai["metrics_primary"]["nominal"],
            "uncompensated": res_ai["metrics_primary"]["deformed_uncompensated"],
            "ridge_blind": res_ridge["metrics_primary"]["healed_quantized"],
            "ai_blind": res_ai["metrics_primary"]["healed_quantized"],
            "oracle": res_oracle["metrics_primary"]["healed_quantized"],
        },
        "uv_vals": u_vals,
        "uv_uncomp": uv_uncomp,
        "uv_healed": uv_healed,
    }


def _grid_triangles(nx: int, ny: int) -> tuple[list[int], list[int], list[int]]:
    """Build deterministic triangle vertex lists (i, j, k) for an (nx, ny) structured grid."""
    ti, tj, tk = [], [], []
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            p00 = ix * ny + iy
            p10 = (ix + 1) * ny + iy
            p11 = (ix + 1) * ny + (iy + 1)
            p01 = ix * ny + (iy + 1)
            ti.extend([p00, p00])
            tj.extend([p10, p11])
            tk.extend([p11, p01])
    return ti, tj, tk


def make_3d_aperture_figure(
    wing_cfg: WingboxConfig,
    array_cfg: ArrayConfig,
    fbg_cfg: FBGSensorConfig,
    true_disp: np.ndarray,
    pred_disp: np.ndarray,
    dead_sensor_ids: Sequence[int],
    detected_class_name: str,
) -> go.Figure:
    """Build 3D UAV wingbox frame + deformed AESA aperture + 12 FBG + damage indicator."""
    fig = go.Figure()

    # 1. Wingbox structural skeleton at z = z_top
    zt = wing_cfg.z_top
    L, c = wing_cfg.span, wing_cfg.chord
    outline_x = [0, L, L, 0, 0, None, 0, L, None, 0, L]
    outline_y = [0, 0, c, c, 0, None, wing_cfg.y_fspar, wing_cfg.y_fspar, None, wing_cfg.y_rspar, wing_cfg.y_rspar]
    outline_z = [zt] * len(outline_x)
    for rx in wing_cfg.rib_x[1:]:
        outline_x.extend([None, rx, rx])
        outline_y.extend([None, 0.0, c])
        outline_z.extend([None, zt, zt])

    fig.add_trace(
        go.Scatter3d(
            x=outline_x,
            y=outline_y,
            z=outline_z,
            mode="lines",
            line=dict(color="#475569", width=3),
            name="Wingbox Spars & Ribs",
            hoverinfo="skip",
        )
    )

    # 2. Highlight detected battle-damage zone in Bay 2 (x in [0.30, 0.60])
    if detected_class_name == "front_spar_bay2":
        fig.add_trace(
            go.Scatter3d(
                x=[0.30, 0.60],
                y=[wing_cfg.y_fspar, wing_cfg.y_fspar],
                z=[zt, zt],
                mode="lines",
                line=dict(color="#FF4D6D", width=9),
                name="DETECTED DAMAGE: Front Spar Bay 2",
            )
        )
    elif detected_class_name == "rear_spar_bay2":
        fig.add_trace(
            go.Scatter3d(
                x=[0.30, 0.60],
                y=[wing_cfg.y_rspar, wing_cfg.y_rspar],
                z=[zt, zt],
                mode="lines",
                line=dict(color="#FF4D6D", width=9),
                name="DETECTED DAMAGE: Rear Spar Bay 2",
            )
        )
    elif detected_class_name == "upper_skin_bay2":
        fig.add_trace(
            go.Mesh3d(
                x=[0.30, 0.60, 0.60, 0.30],
                y=[wing_cfg.y_fspar, wing_cfg.y_fspar, wing_cfg.y_rspar, wing_cfg.y_rspar],
                z=[zt - 0.001] * 4,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color="#FF4D6D",
                opacity=0.35,
                name="DETECTED DAMAGE: Upper Skin Bay 2",
            )
        )

    # 3. Deformed 16x16 Conformal AESA Aperture (2.5x visual Z scale for crystal clarity)
    exag = 2.5
    pos_nom = array_cfg.nominal_element_positions()
    pos_true_vis = pos_nom + exag * true_disp
    pos_ai_vis = pos_nom + exag * pred_disp
    uz_mm = true_disp[:, 2] * 1e3
    ti, tj, tk = _grid_triangles(array_cfg.nx, array_cfg.ny)

    fig.add_trace(
        go.Mesh3d(
            x=pos_true_vis[:, 0],
            y=pos_true_vis[:, 1],
            z=pos_true_vis[:, 2],
            i=ti,
            j=tj,
            k=tk,
            intensity=uz_mm,
            colorscale="Turbo",
            colorbar=dict(title="True UZ (mm)", len=0.75, x=1.02),
            opacity=0.88,
            name="True ANSYS Aperture (2.5x Z)",
            hovertemplate="x=%{x:.3f}m<br>y=%{y:.3f}m<br>UZ=%{intensity:.2f}mm<extra></extra>",
        )
    )

    # Sparse subset of AI-predicted markers so users visually see AI tracking the ANSYS skin
    fig.add_trace(
        go.Scatter3d(
            x=pos_ai_vis[::4, 0],
            y=pos_ai_vis[::4, 1],
            z=pos_ai_vis[::4, 2] + 0.002,
            mode="markers",
            marker=dict(size=3.2, color="#06D6A0", symbol="circle-open"),
            name="ShapeNet AI Estimate",
        )
    )

    # 4. 12 FBG Sensors (Active Cyan vs Severed Crimson X)
    fbg_xy = fbg_cfg.as_array()
    dead_set = set(dead_sensor_ids)
    active_idx = [i for i in range(len(fbg_xy)) if i not in dead_set]
    dead_idx = [i for i in range(len(fbg_xy)) if i in dead_set]

    if active_idx:
        fig.add_trace(
            go.Scatter3d(
                x=fbg_xy[active_idx, 0],
                y=fbg_xy[active_idx, 1],
                z=[zt + 0.004] * len(active_idx),
                mode="markers",
                marker=dict(size=6, color="#00F5D4", symbol="diamond"),
                text=[f"FBG-{i + 1} (ONLINE)" for i in active_idx],
                hoverinfo="text",
                name="FBG Online",
            )
        )
    if dead_idx:
        fig.add_trace(
            go.Scatter3d(
                x=fbg_xy[dead_idx, 0],
                y=fbg_xy[dead_idx, 1],
                z=[zt + 0.004] * len(dead_idx),
                mode="markers",
                marker=dict(size=7, color="#FF4D6D", symbol="x"),
                text=[f"FBG-{i + 1} (SEVERED)" for i in dead_idx],
                hoverinfo="text",
                name="FBG Severed",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=10, t=36, b=0),
        title=dict(text="3D Composite Wingbox, Conformal AESA & FBG Network", x=0.02, font=dict(size=14)),
        scene=dict(
            xaxis_title="Span X (m)",
            yaxis_title="Chord Y (m)",
            zaxis_title="Z (2.5x def, m)",
            aspectmode="manual",
            aspectratio=dict(x=1.8, y=0.85, z=0.45),
            camera=dict(eye=dict(x=1.45, y=-1.35, z=0.85)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.06, xanchor="center", x=0.5),
        height=440,
    )
    return fig


def make_1d_pattern_figure(res: Dict[str, Any]) -> go.Figure:
    """Plot 1D X-band pattern cut comparing Nominal, Uncompensated, Ridge, and ShapeNet Blind."""
    th = res["theta_deg"]
    cuts = res["cuts"]
    m = res["metrics"]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=th,
            y=cuts["nominal"],
            mode="lines",
            line=dict(color="#94A3B8", width=1.8, dash="dash"),
            name=f"Nominal (SLL={m['nominal']['sll_db']:.1f} dB)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=th,
            y=cuts["uncompensated"],
            mode="lines",
            line=dict(color="#FF4D6D", width=2.4),
            name=f"Uncompensated (Err={m['uncompensated']['boresight_error_deg']:+.2f}°, SLL={m['uncompensated']['sll_db']:.1f} dB)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=th,
            y=cuts["ridge_blind"],
            mode="lines",
            line=dict(color="#FFD166", width=2.0, dash="dot"),
            name=f"Ridge Blind (Err={m['ridge_blind']['boresight_error_deg']:+.2f}°, SLL={m['ridge_blind']['sll_db']:.1f} dB)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=th,
            y=cuts["ai_blind"],
            mode="lines",
            line=dict(color="#06D6A0", width=2.8),
            name=f"ShapeNet Blind (Err={m['ai_blind']['boresight_error_deg']:+.2f}°, SLL={m['ai_blind']['sll_db']:.1f} dB)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[res["theta0_deg"], res["theta0_deg"]],
            y=[-50.0, 2.0],
            mode="lines",
            line=dict(color="#00F5D4", width=1.5, dash="dashdot"),
            name=f"Target θ₀ = {res['theta0_deg']:+.0f}°",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        margin=dict(l=45, r=15, t=42, b=35),
        title=dict(text="X-Band (10 GHz) Far-Field Radiation Pattern Cut", x=0.02, font=dict(size=14)),
        xaxis=dict(title="Scan Angle θ (deg)", range=[-55, 55]),
        yaxis=dict(title="Normalized Gain (dB)", range=[-48, 3]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.34, xanchor="center", x=0.5, font=dict(size=10)),
        height=440,
    )
    return fig


def make_2d_uv_comparison_figure(res: Dict[str, Any]) -> go.Figure:
    """Plot side-by-side 2D direction-cosine (u, v) radar beam heatmaps."""
    uv = res["uv_vals"]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncompensated 2D Beam (Shattered / Drifted)", "ShapeNet Blind 6-Bit Self-Healed Beam"),
        horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Heatmap(
            x=uv,
            y=uv,
            z=res["uv_uncomp"].T,
            zmin=-45,
            zmax=0,
            colorscale="Inferno",
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=uv,
            y=uv,
            z=res["uv_healed"].T,
            zmin=-45,
            zmax=0,
            colorscale="Inferno",
            colorbar=dict(title="dB", len=0.85),
        ),
        row=1,
        col=2,
    )

    u0 = float(np.sin(np.deg2rad(res["theta0_deg"])))
    for col in (1, 2):
        fig.add_trace(
            go.Scatter(
                x=[u0],
                y=[0.0],
                mode="markers",
                marker=dict(size=11, color="#00F5D4", symbol="cross-thin", line=dict(width=2, color="#00F5D4")),
                showlegend=(col == 1),
                name="Target Lock (u₀, v₀)",
            ),
            row=1,
            col=col,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=48, b=35),
        height=370,
    )
    for col in (1, 2):
        fig.update_xaxes(title_text="u = sinθ cosφ", row=1, col=col)
        fig.update_yaxes(title_text="v = sinθ sinφ", scaleanchor=f"x{'' if col == 1 else '2'}", row=1, col=col)
    return fig


def make_fbg_parity_figure(res: Dict[str, Any]) -> go.Figure:
    """Plot per-sensor parity-space anomaly signature across all 12 FBG gauges."""
    scores = res["sensor_parity_scores"]
    dead_set = set(res["dead_sensor_ids"])
    labels = [f"FBG-{i + 1}" for i in range(len(scores))]

    colors = []
    status_text = []
    for i, val in enumerate(scores):
        if i in dead_set:
            colors.append("#FF4D6D")
            status_text.append("SEVERED (Imputed)")
        elif val > 0.14:
            colors.append("#FFD166")
            status_text.append("Anomaly Spike")
        else:
            colors.append("#00F5D4")
            status_text.append("Healthy")

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=scores,
            marker_color=colors,
            text=[f"{v:.3f}" if i not in dead_set else "DEAD" for i, v in enumerate(scores)],
            textposition="outside",
            hovertext=status_text,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        margin=dict(l=40, r=15, t=42, b=35),
        title=dict(text="12-FBG Parity-Space Strain Residual Signature", x=0.02, font=dict(size=14)),
        yaxis=dict(title="Normalized Parity Residual ||r||₂", range=[0, max(0.35, float(np.max(scores)) * 1.25)]),
        height=370,
    )
    return fig

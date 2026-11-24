"""Generate a 36-frame playable Plotly LinkedIn video-demo HTML for AeroRF-Heal."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerorf.dashboard_core import _grid_triangles, load_cockpit_assets, run_live_combat_inference
from aerorf.rf_array import (
    compute_pattern_cut,
    compute_steering_weights,
    make_2d_taylor_taper,
    rotate_normals_euler,
)


def pick_dramatic_heldout_case(assets) -> int:
    """Select the held-out damaged ANSYS case with largest aperture deflection."""
    best_idx = 0
    best_score = -1.0
    for idx in np.where(assets.heldout_mask)[0]:
        if assets.damage_label[idx] == 0:
            continue
        uz_span = float(assets.array_disp[idx, :, 2].max() - assets.array_disp[idx, :, 2].min())
        uz_max = float(np.abs(assets.array_disp[idx, :, 2]).max())
        score = uz_span + 0.5 * uz_max
        if score > best_score:
            best_score = score
            best_idx = int(idx)
    return best_idx


def main() -> None:
    assets = load_cockpit_assets(ROOT)
    case_idx = pick_dramatic_heldout_case(assets)
    dead_ids = [3, 7]  # FBG-4 & FBG-8 severed in combat
    theta0_deg = 15.0

    live = run_live_combat_inference(
        assets=assets,
        case_idx=case_idx,
        dead_sensor_ids=dead_ids,
        noise_pct=3.0,
        theta0_deg=theta0_deg,
        quant_bits=6,
        rng_seed=777,
    )

    array_cfg = assets.array_cfg
    wing_cfg = assets.wing_cfg
    fbg_cfg = assets.fbg_cfg
    pos_nom = array_cfg.nominal_element_positions()
    norm_nom = array_cfg.nominal_normals()
    disp_true = assets.array_disp[case_idx]
    rot_true = assets.array_rot[case_idx]
    disp_ai = live["disp_ai"]

    amp_weights = make_2d_taylor_taper(
        array_cfg.nx, array_cfg.ny, sll_db=array_cfg.sll_target_db, nbar=array_cfg.nbar
    )
    w_nominal = compute_steering_weights(
        pos_nom, theta0_deg, 0.0, array_cfg.k0, amp_weights, quant_bits=None
    )

    theta_grid = np.linspace(-60.0, 60.0, 721)
    pat_nom, e_nom = compute_pattern_cut(
        pos_nom, norm_nom, w_nominal, array_cfg.k0, theta_grid, phi_deg=0.0, q_elem=array_cfg.q_elem
    )
    ref_peak = float(np.max(e_nom))

    exag = 2.5
    ti, tj, tk = _grid_triangles(array_cfg.nx, array_cfg.ny)

    # Initial frame (t = 0: flat wing)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.50, 0.50],
        subplot_titles=(
            "Conformal 16×16 X-Band Wing Aperture (True ANSYS Deflection, 2.5x Z)",
            f"Live Far-Field Beam Pattern (Target θ₀ = +{theta0_deg:.0f}°, 2 Dead FBGs + 3% Noise)",
        ),
        horizontal_spacing=0.06,
    )

    # Trace 0 (dynamic): 3D Deformed Aperture Mesh
    fig.add_trace(
        go.Mesh3d(
            x=pos_nom[:, 0],
            y=pos_nom[:, 1],
            z=pos_nom[:, 2],
            i=ti,
            j=tj,
            k=tk,
            intensity=np.zeros(len(pos_nom)),
            cmin=-30.0,
            cmax=30.0,
            colorscale="Turbo",
            colorbar=dict(title="UZ (mm)", x=0.45, len=0.75),
            opacity=0.90,
            name="AESA Aperture",
        ),
        row=1,
        col=1,
    )

    # Trace 1 (dynamic): Live X-Band Pattern Cut
    fig.add_trace(
        go.Scatter(
            x=theta_grid,
            y=pat_nom,
            mode="lines",
            line=dict(color="#FF4D6D", width=3.0),
            name="Live Radar Beam",
        ),
        row=1,
        col=2,
    )

    # Trace 2 (static): Nominal reference cut
    fig.add_trace(
        go.Scatter(
            x=theta_grid,
            y=pat_nom,
            mode="lines",
            line=dict(color="#94A3B8", width=1.8, dash="dash"),
            name="Nominal Ideal (-30 dB Taylor)",
        ),
        row=1,
        col=2,
    )

    # Trace 3 (static): Target angle vertical marker
    fig.add_trace(
        go.Scatter(
            x=[theta0_deg, theta0_deg],
            y=[-48.0, 2.5],
            mode="lines",
            line=dict(color="#00F5D4", width=1.6, dash="dot"),
            name=f"Target +{theta0_deg:.0f}°",
        ),
        row=1,
        col=2,
    )

    # Trace 4 & 5 (static): Wingbox frame + 12 FBG status pins
    zt = wing_cfg.z_top
    L, c = wing_cfg.span, wing_cfg.chord
    ox = [0, L, L, 0, 0, None, 0, L, None, 0, L]
    oy = [0, 0, c, c, 0, None, wing_cfg.y_fspar, wing_cfg.y_fspar, None, wing_cfg.y_rspar, wing_cfg.y_rspar]
    oz = [zt] * len(ox)
    fig.add_trace(
        go.Scatter3d(
            x=ox, y=oy, z=oz, mode="lines", line=dict(color="#475569", width=3), name="Wingbox Frame"
        ),
        row=1,
        col=1,
    )

    fbg_xy = fbg_cfg.as_array()
    act_idx = [i for i in range(len(fbg_xy)) if i not in dead_ids]
    fig.add_trace(
        go.Scatter3d(
            x=fbg_xy[act_idx, 0],
            y=fbg_xy[act_idx, 1],
            z=[zt + 0.004] * len(act_idx),
            mode="markers",
            marker=dict(size=5, color="#00F5D4", symbol="diamond"),
            name="10 Active FBGs",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=fbg_xy[dead_ids, 0],
            y=fbg_xy[dead_ids, 1],
            z=[zt + 0.004] * len(dead_ids),
            mode="markers",
            marker=dict(size=7, color="#FF4D6D", symbol="x"),
            name="2 Severed FBGs",
        ),
        row=1,
        col=1,
    )

    # Build 36 frames:
    # Frames 0..17: Wing bends 0 -> 100%, AI healing OFF
    # Frames 18..35: Wing stays at 100% deflection, AI healing ramps 0 -> 100% (6-bit)
    frames = []
    slider_steps = []
    n_bend = 18
    n_heal = 18
    total_frames = n_bend + n_heal

    for f_idx in range(total_frames):
        if f_idx < n_bend:
            alpha_bend = f_idx / float(n_bend - 1)
            alpha_heal = 0.0
            phase_label = f"PHASE 1 — 5G + Thermal Flex ({alpha_bend * 100:.0f}%) | AI Healing: OFF"
            beam_color = "#FF4D6D"
        else:
            alpha_bend = 1.0
            alpha_heal = (f_idx - n_bend + 1) / float(n_heal)
            phase_label = (
                f"PHASE 2 — AI Self-Healing Engaged ({alpha_heal * 100:.0f}%, 0.8 ms) | "
                f"Detected: {live['pred_class']}"
            )
            beam_color = "#06D6A0" if alpha_heal > 0.5 else "#FFD166"

        cur_disp = alpha_bend * disp_true
        cur_rot = alpha_bend * rot_true
        cur_pos = pos_nom + cur_disp
        cur_norm = rotate_normals_euler(norm_nom, cur_rot)

        # Blend estimated compensation displacement smoothly from 0 -> disp_ai
        cur_est = pos_nom + alpha_heal * disp_ai
        cur_w = compute_steering_weights(
            cur_est,
            theta0_deg,
            0.0,
            array_cfg.k0,
            amp_weights,
            quant_bits=6 if alpha_heal > 0 else None,
        )
        cur_pat, _ = compute_pattern_cut(
            cur_pos,
            cur_norm,
            cur_w,
            array_cfg.k0,
            theta_grid,
            phi_deg=0.0,
            q_elem=array_cfg.q_elem,
            ref_peak_linear=ref_peak,
        )

        pos_vis = pos_nom + exag * cur_disp
        uz_cur_mm = cur_disp[:, 2] * 1e3
        fname = f"f{f_idx:02d}"

        frames.append(
            go.Frame(
                name=fname,
                traces=[0, 1],
                data=[
                    go.Mesh3d(
                        x=pos_vis[:, 0],
                        y=pos_vis[:, 1],
                        z=pos_vis[:, 2],
                        intensity=uz_cur_mm,
                    ),
                    go.Scatter(
                        y=cur_pat,
                        line=dict(color=beam_color, width=3.0),
                    ),
                ],
                layout=go.Layout(title_text=f"AeroRF-Heal Live Demo (Held-Out Case #{case_idx}) — {phase_label}"),
            )
        )
        slider_steps.append(
            dict(
                args=[[fname], dict(frame=dict(duration=90, redraw=True), mode="immediate", transition=dict(duration=0))],
                label=f"{f_idx + 1}",
                method="animate",
            )
        )

    fig.frames = frames
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"AeroRF-Heal Live Demo (Held-Out Case #{case_idx}, {live['true_class']}) — Press Play",
            x=0.5,
            font=dict(size=16),
        ),
        width=1400,
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="center", x=0.5),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=-0.08,
                buttons=[
                    dict(
                        label="▶ Play Demo",
                        method="animate",
                        args=[
                            None,
                            dict(frame=dict(duration=110, redraw=True), fromcurrent=True, transition=dict(duration=0)),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                x=0.18,
                len=0.78,
                y=-0.08,
                pad=dict(b=10, t=10),
                currentvalue=dict(prefix="Frame: ", visible=True),
            )
        ],
    )
    fig.update_scenes(
        xaxis_title="Span X (m)",
        yaxis_title="Chord Y (m)",
        zaxis_title="Z (2.5x def, m)",
        zaxis=dict(range=[-0.03, 0.14]),
        aspectmode="manual",
        aspectratio=dict(x=1.8, y=0.85, z=0.55),
        camera=dict(eye=dict(x=1.45, y=-1.35, z=0.85)),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Scan Angle θ (deg)", range=[-55, 55], row=1, col=2)
    fig.update_yaxes(title_text="Normalized Gain (dB)", range=[-48, 3], row=1, col=2)

    out_html = ROOT / "outputs" / "linkedin_demo_animation.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", auto_play=False)
    size_kb = out_html.stat().st_size / 1024.0
    print(f"Generated LinkedIn 36-frame demo ({size_kb:.1f} KB): {out_html}")
    print(
        f"Selected Held-Out Case #{case_idx} ({live['true_class']}, df={live['true_severity']:.2f}): "
        f"Boresight {live['metrics']['uncompensated']['boresight_error_deg']:+.2f}° -> "
        f"{live['metrics']['ai_blind']['boresight_error_deg']:+.2f}° | "
        f"SLL {live['metrics']['uncompensated']['sll_db']:.1f} -> {live['metrics']['ai_blind']['sll_db']:.1f} dB"
    )


if __name__ == "__main__":
    main()

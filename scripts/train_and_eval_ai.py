"""Train and benchmark AeroRFShapeNet vs RidgeBaseline on real ANSYS DOE data."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerorf.config import ArrayConfig
from aerorf.rf_array import evaluate_conformal_scenario
from aerorf.shape_ai import (
    RidgeBaseline,
    apply_combat_sensor_augmentation,
    predict_with_latency,
    save_checkpoint,
    train_shapenet,
)


def rmse_mm(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)) * 1.0e3)


def uz_rmse_mm(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred[:, :, 2] - true[:, :, 2]) ** 2)) * 1.0e3)


def eval_regime(
    name: str,
    model,
    pod,
    ridge: RidgeBaseline,
    strains_test_norm: np.ndarray,
    disp_test: np.ndarray,
    lbl_test: np.ndarray,
    sev_test: np.ndarray,
    noise_std: float,
    dropouts: int,
    n_mc: int = 1,
    seed: int = 101,
) -> Dict[str, object]:
    """Evaluate AI & Ridge across one combat sensor stress regime."""
    rng = np.random.default_rng(seed)
    ai_rmses, uz_ai_rmses, ridge_rmses, uz_ridge_rmses = [], [], [], []
    cls_accs, sev_maes = [], []
    all_true_lbl, all_pred_lbl = [], []

    last_disp_ai = None
    last_disp_ridge = None
    last_latency = 0.0

    for _ in range(n_mc):
        x_48, s36, _ = apply_combat_sensor_augmentation(
            strains_test_norm,
            noise_std=noise_std,
            max_dropouts=dropouts,
            exact_dropouts=True,
            rng=rng,
        )
        disp_ai, cls_pred, sev_pred, latency_ms = predict_with_latency(model, pod, x_48)
        disp_ridge = ridge.predict(s36)

        ai_rmses.append(rmse_mm(disp_ai, disp_test))
        uz_ai_rmses.append(uz_rmse_mm(disp_ai, disp_test))
        ridge_rmses.append(rmse_mm(disp_ridge, disp_test))
        uz_ridge_rmses.append(uz_rmse_mm(disp_ridge, disp_test))
        cls_accs.append(float(accuracy_score(lbl_test, cls_pred)))
        sev_maes.append(float(np.mean(np.abs(sev_pred - sev_test))))

        all_true_lbl.append(lbl_test)
        all_pred_lbl.append(cls_pred)
        last_disp_ai = disp_ai
        last_disp_ridge = disp_ridge
        last_latency = latency_ms

    return {
        "name": name,
        "noise_std": noise_std,
        "dead_sensors": dropouts,
        "ai_rmse_mm": float(np.mean(ai_rmses)),
        "ai_uz_rmse_mm": float(np.mean(uz_ai_rmses)),
        "ridge_rmse_mm": float(np.mean(ridge_rmses)),
        "ridge_uz_rmse_mm": float(np.mean(uz_ridge_rmses)),
        "damage_acc_pct": float(np.mean(cls_accs) * 100.0),
        "severity_mae": float(np.mean(sev_maes)),
        "latency_ms": float(last_latency),
        "true_labels": np.concatenate(all_true_lbl),
        "pred_labels": np.concatenate(all_pred_lbl),
        "sample_disp_ai": last_disp_ai,
        "sample_disp_ridge": last_disp_ridge,
    }


def build_benchmark_html(
    html_path: Path,
    regimes: List[Dict[str, object]],
    class_names: List[str],
    rf_summary: Dict[str, Dict[str, float]],
    worst_rf_example: Dict[str, object],
) -> Path:
    """Create interactive Phase 2 benchmark & closed-loop RF recovery HTML report."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "3D Displacement RMSE (mm): Ridge vs AeroRFShapeNet",
            "Battle-Damage Confusion Matrix (3% Noise + 2 Dead FBGs)",
            f"Worst Combat Case Blind Self-Healing ({worst_rf_example['label']})",
        ),
        column_widths=[0.32, 0.30, 0.38],
        horizontal_spacing=0.07,
    )

    # Panel 1: RMSE across regimes
    reg_names = [r["name"] for r in regimes]
    fig.add_trace(
        go.Bar(
            name="Linear Ridge Baseline",
            x=reg_names,
            y=[r["ridge_rmse_mm"] for r in regimes],
            marker_color="#FF4D6D",
            text=[f"{r['ridge_rmse_mm']:.2f}" for r in regimes],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="AeroRFShapeNet (POD+Physics)",
            x=reg_names,
            y=[r["ai_rmse_mm"] for r in regimes],
            marker_color="#06D6A0",
            text=[f"{r['ai_rmse_mm']:.2f}" for r in regimes],
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    # Panel 2: Confusion Matrix on Scenario C
    scen_c = regimes[-1]
    cm = confusion_matrix(
        scen_c["true_labels"],
        scen_c["pred_labels"],
        labels=list(range(len(class_names))),
    )
    short_names = [c.replace("_bay2", "").replace("_", " ") for c in class_names]
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=short_names,
            y=short_names,
            colorscale="Tealgrn",
            text=cm,
            texttemplate="%{text}",
            showscale=False,
        ),
        row=1,
        col=2,
    )

    # Panel 3: Worst-case blind Scenario C RF pattern recovery
    th = worst_rf_example["theta_deg"]
    for label, key, color, dash in (
        ("Nominal", "nominal", "#A8B2C1", "dash"),
        ("Uncompensated", "uncomp", "#FF4D6D", "solid"),
        ("Ridge Blind 6-Bit", "ridge", "#FFD166", "dot"),
        ("AeroRFShapeNet Blind 6-Bit", "ai", "#06D6A0", "solid"),
    ):
        fig.add_trace(
            go.Scatter(
                x=th,
                y=worst_rf_example[key],
                mode="lines",
                line=dict(color=color, width=2.4 if key in ("ai", "uncomp") else 2.0, dash=dash),
                name=label,
            ),
            row=1,
            col=3,
        )

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        title=dict(
            text=(
                f"AeroRF-Heal Phase 2 AI Benchmark — "
                f"Combat (2 Dead FBGs + 3% Noise) Blind SLL: "
                f"Uncomp {rf_summary['uncompensated']['mean_sll_db']:.1f} dB → "
                f"AI {rf_summary['ai_blind_6bit']['mean_sll_db']:.1f} dB | "
                f"Latency {scen_c['latency_ms']:.2f} ms"
            ),
            x=0.5,
            font=dict(size=15),
        ),
        height=640,
        width=1600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.20, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="3D Aperture RMSE (mm)", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Class", row=1, col=2)
    fig.update_yaxes(title_text="True Class", autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="Scan Angle θ (deg)", range=[-60, 60], row=1, col=3)
    fig.update_yaxes(title_text="Normalized Gain (dB)", range=[-48, 3], row=1, col=3)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    return html_path


def main() -> None:
    dataset_path = ROOT / "data" / "aerorf_doe_dataset.npz"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing DOE dataset: {dataset_path}")

    raw = np.load(dataset_path, allow_pickle=False)
    fbg_strains = raw["fbg_strains"]
    array_disp = raw["array_disp"]
    array_rot = raw["array_rot"]
    damage_label = raw["damage_label"]
    damage_factor = raw["damage_factor"]
    class_names = [str(c) for c in raw["class_names"]]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, te_idx = next(splitter.split(fbg_strains, damage_label))

    fbg_tr, fbg_te = fbg_strains[tr_idx], fbg_strains[te_idx]
    disp_tr, disp_te = array_disp[tr_idx], array_disp[te_idx]
    rot_te = array_rot[te_idx]
    lbl_tr, lbl_te = damage_label[tr_idx], damage_label[te_idx]
    sev_tr, sev_te = damage_factor[tr_idx], damage_factor[te_idx]

    array_cfg = ArrayConfig()

    print(f"Loaded {len(damage_label)} real ANSYS DOE cases ({len(tr_idx)} train / {len(te_idx)} held-out test).")

    # 1. Fit Linear Ridge Baseline
    ridge = RidgeBaseline().fit(fbg_tr, disp_tr)

    # 2. Train Physics-Informed POD AeroRFShapeNet
    model, pod, strain_mean, strain_std, hist = train_shapenet(
        fbg_tr,
        disp_tr,
        lbl_tr,
        sev_tr,
        array_cfg=array_cfg,
        max_modes=12,
        hidden_dim=128,
        epochs=350,
        batch_size=16,
        lr=1.5e-3,
        seed=42,
    )
    print(
        f"Trained AeroRFShapeNet on {hist.device_used} in {hist.train_seconds:.2f}s | "
        f"Best epoch {hist.best_epoch}/{hist.epochs} | POD energy ({pod.n_modes} modes): {pod.energy_captured()*100:.3f}%"
    )

    ckpt_path = ROOT / "models" / "aerorf_shapenet.pt"
    save_checkpoint(ckpt_path, model, pod, strain_mean, strain_std)

    strains_te_norm = (fbg_te - strain_mean) / strain_std

    # 3. Stress-test across 3 sensor regimes
    regimes = [
        eval_regime(
            "A: Clean (12/12)",
            model, pod, ridge, strains_te_norm, disp_te, lbl_te, sev_te,
            noise_std=0.0, dropouts=0, n_mc=1, seed=101,
        ),
        eval_regime(
            "B: 3% Noise (12/12)",
            model, pod, ridge, strains_te_norm, disp_te, lbl_te, sev_te,
            noise_std=0.03, dropouts=0, n_mc=5, seed=202,
        ),
        eval_regime(
            "C: Combat (3% + 2 Dead)",
            model, pod, ridge, strains_te_norm, disp_te, lbl_te, sev_te,
            noise_std=0.03, dropouts=2, n_mc=5, seed=303,
        ),
    ]

    print("\n=== [1] HELD-OUT TEST BENCHMARK (24 REAL ANSYS CASES) ===")
    print(
        f"{'Regime':<24} {'Ridge RMSE':>12} {'AI RMSE':>10} {'AI UZ RMSE':>12} "
        f"{'Dmg Acc':>9} {'Sev MAE':>9} {'Latency':>10}"
    )
    print("-" * 92)
    for r in regimes:
        print(
            f"{r['name']:<24} {r['ridge_rmse_mm']:>9.3f} mm {r['ai_rmse_mm']:>7.3f} mm "
            f"{r['ai_uz_rmse_mm']:>9.3f} mm {r['damage_acc_pct']:>8.1f}% "
            f"{r['severity_mae']:>9.3f} {r['latency_ms']:>7.3f} ms"
        )

    # 4. Closed-Loop Blind Self-Healing RF Recovery on Scenario C (single fixed combat draw)
    x_c_48, s36_c, _ = apply_combat_sensor_augmentation(
        strains_te_norm, noise_std=0.03, max_dropouts=2, exact_dropouts=True, rng=np.random.default_rng(777)
    )
    disp_ai_c, _, _, _ = predict_with_latency(model, pod, x_c_48)
    disp_ridge_c = ridge.predict(s36_c)

    rf_buckets: Dict[str, List[Dict[str, float]]] = {
        "uncompensated": [],
        "ridge_blind_6bit": [],
        "ai_blind_6bit": [],
        "oracle_6bit": [],
    }

    worst_idx = 0
    worst_uncomp_score = -1.0
    worst_rf_example: Dict[str, object] = {}

    for i in range(len(te_idx)):
        res_ai = evaluate_conformal_scenario(
            array_cfg,
            disp_te[i],
            rot_te[i],
            theta0_deg=15.0,
            phi0_deg=0.0,
            est_disp_xyz=disp_ai_c[i],
            quant_bits=6,
        )
        res_ridge = evaluate_conformal_scenario(
            array_cfg,
            disp_te[i],
            rot_te[i],
            theta0_deg=15.0,
            phi0_deg=0.0,
            est_disp_xyz=disp_ridge_c[i],
            quant_bits=6,
        )
        res_oracle = evaluate_conformal_scenario(
            array_cfg,
            disp_te[i],
            rot_te[i],
            theta0_deg=15.0,
            phi0_deg=0.0,
            est_disp_xyz=disp_te[i],
            quant_bits=6,
        )

        m_u = res_ai["metrics_primary"]["deformed_uncompensated"]
        m_a = res_ai["metrics_primary"]["healed_quantized"]
        m_r = res_ridge["metrics_primary"]["healed_quantized"]
        m_o = res_oracle["metrics_primary"]["healed_quantized"]

        rf_buckets["uncompensated"].append(m_u)
        rf_buckets["ridge_blind_6bit"].append(m_r)
        rf_buckets["ai_blind_6bit"].append(m_a)
        rf_buckets["oracle_6bit"].append(m_o)

        score = m_u["gain_loss_db"] + abs(m_u["boresight_error_deg"])
        if score > worst_uncomp_score:
            worst_uncomp_score = score
            worst_idx = i
            worst_rf_example = {
                "label": f"{class_names[lbl_te[i]]}, df={sev_te[i]:.2f}",
                "theta_deg": res_ai["theta_deg"],
                "nominal": res_ai["cuts_primary"]["nominal"],
                "uncomp": res_ai["cuts_primary"]["deformed_uncompensated"],
                "ridge": res_ridge["cuts_primary"]["healed_quantized"],
                "ai": res_ai["cuts_primary"]["healed_quantized"],
            }

    rf_summary: Dict[str, Dict[str, float]] = {}
    print("\n=== [2] CLOSED-LOOP BLIND X-BAND RF RECOVERY (SCENARIO C: 3% NOISE + 2 DEAD FBGs, θ₀=+15°) ===")
    print(
        f"{'Method':<22} {'Mean GainLoss':>14} {'Max GainLoss':>13} "
        f"{'Mean |Bore Err|':>16} {'Mean SLL (dB)':>14}"
    )
    print("-" * 84)
    for k, label in (
        ("uncompensated", "Uncompensated"),
        ("ridge_blind_6bit", "Ridge Blind 6-Bit"),
        ("ai_blind_6bit", "ShapeNet Blind 6-Bit"),
        ("oracle_6bit", "ANSYS Oracle 6-Bit"),
    ):
        rows = rf_buckets[k]
        mg = float(np.mean([r["gain_loss_db"] for r in rows]))
        xg = float(np.max([r["gain_loss_db"] for r in rows]))
        mb = float(np.mean([abs(r["boresight_error_deg"]) for r in rows]))
        ms = float(np.mean([r["sll_db"] for r in rows]))
        rf_summary[k] = {
            "mean_gain_loss_db": mg,
            "max_gain_loss_db": xg,
            "mean_abs_boresight_err_deg": mb,
            "mean_sll_db": ms,
        }
        print(f"{label:<22} {mg:>11.3f} dB {xg:>10.3f} dB {mb:>14.3f}° {ms:>14.2f}")

    # 5. Save JSON benchmark + interactive HTML report
    json_path = ROOT / "outputs" / "phase2_ai_benchmark.json"
    json_payload = {
        "n_total": int(len(damage_label)),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "pod_modes": pod.n_modes,
        "pod_energy_captured": pod.energy_captured(),
        "train_seconds": hist.train_seconds,
        "best_epoch": hist.best_epoch,
        "device_used": hist.device_used,
        "regimes": [
            {
                k: v
                for k, v in r.items()
                if k not in ("true_labels", "pred_labels", "sample_disp_ai", "sample_disp_ridge")
            }
            for r in regimes
        ],
        "rf_recovery_scenario_c": rf_summary,
        "worst_case_index": int(worst_idx),
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    html_path = ROOT / "outputs" / "phase2_ai_report.html"
    build_benchmark_html(html_path, regimes, class_names, rf_summary, worst_rf_example)

    print(f"\nSaved model checkpoint : {ckpt_path}")
    print(f"Saved JSON benchmark   : {json_path}")
    print(f"Saved HTML report      : {html_path}")


if __name__ == "__main__":
    main()

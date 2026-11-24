"""IEEE-exact 3D conformal phased-array RF engine and 6-bit self-healing compensator."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal.windows import taylor

from .config import ArrayConfig


def make_2d_taylor_taper(
    nx: int,
    ny: int,
    sll_db: float = 30.0,
    nbar: int = 4,
) -> np.ndarray:
    """Compute a normalized separable 2D Taylor amplitude taper of shape (nx * ny,)."""
    wx = taylor(nx, nbar=nbar, sll=abs(float(sll_db)), norm=False)
    wy = taylor(ny, nbar=nbar, sll=abs(float(sll_db)), norm=False)
    w2d = np.outer(wx, wy).astype(np.float64).ravel()
    return w2d / np.sum(w2d)


def rotate_normals_euler(normals: np.ndarray, rot_xyz: np.ndarray) -> np.ndarray:
    """Rotate unit normals (N, 3) by extrinsic/intrinsic Euler angles (rx, ry, rz) in radians.

    Applies R = Rz(rz) @ Ry(ry) @ Rx(rx) per element using vectorized trig.
    """
    normals = np.asarray(normals, dtype=np.float64)
    rot_xyz = np.asarray(rot_xyz, dtype=np.float64)

    rx = rot_xyz[:, 0]
    ry = rot_xyz[:, 1]
    rz = rot_xyz[:, 2]

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    nx0, ny0, nz0 = normals[:, 0], normals[:, 1], normals[:, 2]

    # Step 1: Rx(rx)
    x1 = nx0
    y1 = cx * ny0 - sx * nz0
    z1 = sx * ny0 + cx * nz0

    # Step 2: Ry(ry)
    x2 = cy * x1 + sy * z1
    y2 = y1
    z2 = -sy * x1 + cy * z1

    # Step 3: Rz(rz)
    x3 = cz * x2 - sz * y2
    y3 = sz * x2 + cz * y2
    z3 = z2

    out = np.column_stack([x3, y3, z3])
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)


def quantize_phase(phase_rad: np.ndarray, bits: Optional[int]) -> np.ndarray:
    """Quantize phase in radians to 2^bits uniform levels over [0, 2*pi)."""
    if bits is None:
        return np.asarray(phase_rad, dtype=np.float64)
    n_states = 1 << int(bits)
    step = 2.0 * math.pi / n_states
    wrapped = np.mod(phase_rad, 2.0 * math.pi)
    return np.round(wrapped / step) * step


def direction_unit_vector(theta_deg: float, phi_deg: float) -> np.ndarray:
    """Return 3D unit vector u_hat for spherical angles (theta, phi) in degrees."""
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    return np.array(
        [
            math.sin(th) * math.cos(ph),
            math.sin(th) * math.sin(ph),
            math.cos(th),
        ],
        dtype=np.float64,
    )


def compute_steering_weights(
    positions: np.ndarray,
    theta0_deg: float,
    phi0_deg: float,
    k: float,
    amp_weights: np.ndarray,
    quant_bits: Optional[int] = None,
) -> np.ndarray:
    """Compute complex array excitation weights steered to (theta0_deg, phi0_deg)."""
    positions = np.asarray(positions, dtype=np.float64)
    amp_weights = np.asarray(amp_weights, dtype=np.float64)
    u0 = direction_unit_vector(theta0_deg, phi0_deg)
    phase = k * (positions @ u0)
    q_phase = quantize_phase(phase, quant_bits)
    return amp_weights * np.exp(-1j * q_phase)


def compute_pattern_cut(
    positions: np.ndarray,
    normals: np.ndarray,
    weights: np.ndarray,
    k: float,
    theta_deg: np.ndarray,
    phi_deg: float = 0.0,
    q_elem: float = 1.4,
    ref_peak_linear: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate 1D far-field power pattern along theta_deg at fixed phi_deg.

    Returns (pattern_db, e_mag_linear).
    """
    positions = np.asarray(positions, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.complex128)
    theta_deg = np.asarray(theta_deg, dtype=np.float64)

    th_rad = np.deg2rad(theta_deg)
    ph_rad = math.radians(phi_deg)

    # Shape (M, 3) observation unit vectors
    u_vecs = np.column_stack(
        [
            np.sin(th_rad) * math.cos(ph_rad),
            np.sin(th_rad) * math.sin(ph_rad),
            np.cos(th_rad),
        ]
    )

    # Local element cosine factor: shape (N, M)
    cos_local = np.maximum(0.0, normals @ u_vecs.T)
    elem_pat = np.power(cos_local, 0.5 * q_elem)

    # Spatial phase delay: shape (N, M)
    spatial_phase = k * (positions @ u_vecs.T)
    field = np.sum(weights[:, None] * elem_pat * np.exp(1j * spatial_phase), axis=0)
    e_mag = np.abs(field)

    denom = float(ref_peak_linear) if ref_peak_linear is not None else float(np.max(e_mag))
    denom = max(denom, 1e-12)
    pattern_db = 20.0 * np.log10(np.maximum(e_mag / denom, 1e-12))
    return pattern_db, e_mag


def compute_pattern_uv_grid(
    positions: np.ndarray,
    normals: np.ndarray,
    weights: np.ndarray,
    k: float,
    u_vals: np.ndarray,
    v_vals: np.ndarray,
    q_elem: float = 1.4,
    ref_peak_linear: Optional[float] = None,
) -> np.ndarray:
    """Evaluate 2D far-field power pattern (dB) on direction-cosine (u, v) grid."""
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")
    r2 = uu * uu + vv * vv
    visible = r2 <= 1.0
    ww = np.sqrt(np.maximum(0.0, 1.0 - r2))

    u_flat = np.column_stack([uu[visible], vv[visible], ww[visible]])
    cos_local = np.maximum(0.0, normals @ u_flat.T)
    elem_pat = np.power(cos_local, 0.5 * q_elem)
    spatial_phase = k * (positions @ u_flat.T)
    e_mag_vis = np.abs(
        np.sum(weights[:, None] * elem_pat * np.exp(1j * spatial_phase), axis=0)
    )

    denom = (
        float(ref_peak_linear)
        if ref_peak_linear is not None
        else max(float(np.max(e_mag_vis)), 1e-12)
    )
    out_db = np.full(uu.shape, -80.0, dtype=np.float64)
    out_db[visible] = 20.0 * np.log10(np.maximum(e_mag_vis / denom, 1e-12))
    return out_db


def _parabolic_peak(theta_deg: np.ndarray, pattern_db: np.ndarray, idx: int) -> float:
    """Refine discrete grid peak angle with 3-point parabolic interpolation."""
    if idx <= 0 or idx >= len(theta_deg) - 1:
        return float(theta_deg[idx])
    y0, y1, y2 = pattern_db[idx - 1], pattern_db[idx], pattern_db[idx + 1]
    denom = 2.0 * (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-12:
        return float(theta_deg[idx])
    delta = (y0 - y2) / denom
    step = float(theta_deg[idx + 1] - theta_deg[idx])
    return float(theta_deg[idx] + np.clip(delta, -0.5, 0.5) * step)


def _find_null_index(rel_db: np.ndarray, start_idx: int, direction: int) -> int:
    """Walk left (-1) or right (+1) from main-lobe peak to first local minimum (null)."""
    n = len(rel_db)
    idx = start_idx + direction
    while 0 < idx < n - 1:
        if rel_db[idx] <= rel_db[idx - 1] and rel_db[idx] <= rel_db[idx + 1]:
            return idx
        idx += direction
    return 0 if direction < 0 else n - 1


def _find_crossing(
    theta_deg: np.ndarray,
    rel_db: np.ndarray,
    start_idx: int,
    direction: int,
    level_db: float = -3.0,
) -> float:
    """Interpolate angle where pattern crosses level_db on left/right of main peak."""
    n = len(rel_db)
    idx = start_idx
    while 0 <= idx + direction < n:
        nxt = idx + direction
        if rel_db[idx] >= level_db > rel_db[nxt]:
            frac = (level_db - rel_db[idx]) / (rel_db[nxt] - rel_db[idx])
            return float(theta_deg[idx] + frac * (theta_deg[nxt] - theta_deg[idx]))
        idx = nxt
    return float(theta_deg[0 if direction < 0 else n - 1])


def evaluate_pattern_metrics(
    theta_deg: np.ndarray,
    pattern_db: np.ndarray,
    target_theta_deg: float,
) -> Dict[str, float]:
    """Extract peak gain, boresight error, HPBW, and peak SLL from a 1D pattern cut."""
    theta_deg = np.asarray(theta_deg, dtype=np.float64)
    pattern_db = np.asarray(pattern_db, dtype=np.float64)

    peak_idx = int(np.argmax(pattern_db))
    peak_gain_db = float(pattern_db[peak_idx])
    peak_theta_deg = _parabolic_peak(theta_deg, pattern_db, peak_idx)
    boresight_error_deg = peak_theta_deg - float(target_theta_deg)

    rel_db = pattern_db - peak_gain_db
    left_3db = _find_crossing(theta_deg, rel_db, peak_idx, -1, level_db=-3.0)
    right_3db = _find_crossing(theta_deg, rel_db, peak_idx, +1, level_db=-3.0)
    hpbw_deg = float(right_3db - left_3db)

    left_null = _find_null_index(rel_db, peak_idx, -1)
    right_null = _find_null_index(rel_db, peak_idx, +1)

    sidelobe_candidates = []
    if left_null > 0:
        sidelobe_candidates.append(float(np.max(rel_db[: left_null + 1])))
    if right_null < len(rel_db) - 1:
        sidelobe_candidates.append(float(np.max(rel_db[right_null:])))
    sll_db = max(sidelobe_candidates) if sidelobe_candidates else -80.0

    return {
        "peak_gain_db": peak_gain_db,
        "gain_loss_db": -peak_gain_db,
        "peak_theta_deg": peak_theta_deg,
        "boresight_error_deg": boresight_error_deg,
        "hpbw_deg": hpbw_deg,
        "sll_db": sll_db,
    }


def evaluate_conformal_scenario(
    array_cfg: ArrayConfig,
    disp_xyz: np.ndarray,
    rot_xyz: np.ndarray,
    theta0_deg: float = 0.0,
    phi0_deg: float = 0.0,
    est_disp_xyz: Optional[np.ndarray] = None,
    quant_bits: Optional[int] = None,
    n_theta: int = 3601,
) -> Dict[str, object]:
    """Compare nominal, uncompensated deformed, 6-bit healed, and ideal healed patterns.

    Evaluates principal cut along steering plane (phi=phi0_deg) and orthogonal cut
    (phi=phi0_deg + 90 deg) when theta0 == 0, plus azimuth (phi=0) & elevation (phi=90) cuts.
    """
    disp_xyz = np.asarray(disp_xyz, dtype=np.float64)
    rot_xyz = np.asarray(rot_xyz, dtype=np.float64)
    est_disp = disp_xyz if est_disp_xyz is None else np.asarray(est_disp_xyz, dtype=np.float64)
    bits = array_cfg.quant_bits if quant_bits is None else quant_bits

    pos_nom = array_cfg.nominal_element_positions()
    norm_nom = array_cfg.nominal_normals()
    pos_def = pos_nom + disp_xyz
    pos_est = pos_nom + est_disp
    norm_def = rotate_normals_euler(norm_nom, rot_xyz)

    amp_weights = make_2d_taylor_taper(
        array_cfg.nx,
        array_cfg.ny,
        sll_db=array_cfg.sll_target_db,
        nbar=array_cfg.nbar,
    )

    w_nominal = compute_steering_weights(
        pos_nom, theta0_deg, phi0_deg, array_cfg.k0, amp_weights, quant_bits=None
    )
    w_healed_q = compute_steering_weights(
        pos_est, theta0_deg, phi0_deg, array_cfg.k0, amp_weights, quant_bits=bits
    )
    w_healed_ideal = compute_steering_weights(
        pos_est, theta0_deg, phi0_deg, array_cfg.k0, amp_weights, quant_bits=None
    )

    theta_deg = np.linspace(-90.0, 90.0, n_theta)

    # Reference peak from undeformed nominal array at steering angle in phi0 plane
    _, e_nom_ref = compute_pattern_cut(
        pos_nom, norm_nom, w_nominal, array_cfg.k0, theta_deg, phi_deg=phi0_deg, q_elem=array_cfg.q_elem
    )
    ref_peak = float(np.max(e_nom_ref))

    variants = {
        "nominal": (pos_nom, norm_nom, w_nominal),
        "deformed_uncompensated": (pos_def, norm_def, w_nominal),
        "healed_quantized": (pos_def, norm_def, w_healed_q),
        "healed_ideal": (pos_def, norm_def, w_healed_ideal),
    }

    cuts_az = {}
    cuts_el = {}
    metrics_az = {}
    metrics_el = {}

    el_target = theta0_deg if abs(math.sin(math.radians(phi0_deg))) > 0.707 else 0.0
    az_target = theta0_deg if abs(math.cos(math.radians(phi0_deg))) >= 0.707 else 0.0

    for key, (pos, norm, w) in variants.items():
        pat_az, _ = compute_pattern_cut(
            pos, norm, w, array_cfg.k0, theta_deg, phi_deg=0.0, q_elem=array_cfg.q_elem, ref_peak_linear=ref_peak
        )
        pat_el, _ = compute_pattern_cut(
            pos, norm, w, array_cfg.k0, theta_deg, phi_deg=90.0, q_elem=array_cfg.q_elem, ref_peak_linear=ref_peak
        )
        cuts_az[key] = pat_az
        cuts_el[key] = pat_el
        metrics_az[key] = evaluate_pattern_metrics(theta_deg, pat_az, target_theta_deg=az_target)
        metrics_el[key] = evaluate_pattern_metrics(theta_deg, pat_el, target_theta_deg=el_target)

    # Primary cut matching phi0_deg directly
    cuts_primary = {}
    metrics_primary = {}
    for key, (pos, norm, w) in variants.items():
        pat_p, _ = compute_pattern_cut(
            pos, norm, w, array_cfg.k0, theta_deg, phi_deg=phi0_deg, q_elem=array_cfg.q_elem, ref_peak_linear=ref_peak
        )
        cuts_primary[key] = pat_p
        metrics_primary[key] = evaluate_pattern_metrics(theta_deg, pat_p, target_theta_deg=theta0_deg)

    return {
        "theta_deg": theta_deg,
        "theta0_deg": theta0_deg,
        "phi0_deg": phi0_deg,
        "quant_bits": bits,
        "ref_peak_linear": ref_peak,
        "positions_nominal": pos_nom,
        "positions_deformed": pos_def,
        "normals_deformed": norm_def,
        "weights": {
            "nominal": w_nominal,
            "healed_quantized": w_healed_q,
            "healed_ideal": w_healed_ideal,
        },
        "cuts_primary": cuts_primary,
        "metrics_primary": metrics_primary,
        "cuts_az": cuts_az,
        "metrics_az": metrics_az,
        "cuts_el": cuts_el,
        "metrics_el": metrics_el,
    }

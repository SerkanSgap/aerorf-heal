"""Physics-Informed POD Inverse Shape-Sensing & Battle-Damage AI (AeroRFShapeNet v2)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ansys_runner import DAMAGE_CLASSES
from .config import ArrayConfig


def select_torch_device() -> torch.device:
    """Safely select CUDA only if a real tensor matmul succeeds on the GPU, else CPU."""
    if torch.cuda.is_available():
        try:
            a = torch.ones((4, 4), device="cuda")
            _ = float((a @ a).sum().cpu())
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


class PODBasis:
    """Proper Orthogonal Decomposition (SVD) modal basis for 256x3 aperture displacements."""

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None  # (n_modes, 768)
        self.singular_values: Optional[np.ndarray] = None
        self.captured_energy: float = 0.0

    @property
    def n_modes(self) -> int:
        if self.components is None:
            return 0
        return int(self.components.shape[0])

    def fit(self, disp_xyz: np.ndarray, max_modes: int = 12) -> "PODBasis":
        """Fit SVD basis on displacements of shape (N, n_elem, 3) or (N, 3*n_elem)."""
        flat = np.asarray(disp_xyz, dtype=np.float64).reshape(len(disp_xyz), -1)
        self.mean = flat.mean(axis=0)
        centered = flat - self.mean
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(int(max_modes), len(s))
        self.components = vt[:k].copy()
        self.singular_values = s.copy()
        total_energy = float(np.sum(s**2))
        self.captured_energy = (
            float(np.sum(s[:k] ** 2)) / total_energy if total_energy > 1e-18 else 1.0
        )
        return self

    def energy_captured(self) -> float:
        return self.captured_energy

    def encode(self, disp_xyz: np.ndarray) -> np.ndarray:
        """Project displacements onto POD modal coordinates of shape (N, n_modes)."""
        assert self.mean is not None and self.components is not None
        flat = np.asarray(disp_xyz, dtype=np.float64).reshape(len(disp_xyz), -1)
        return (flat - self.mean) @ self.components.T

    def decode_numpy(self, q: np.ndarray) -> np.ndarray:
        """Reconstruct 3D displacements (N, n_elem, 3) from modal coordinates q."""
        assert self.mean is not None and self.components is not None
        q_arr = np.asarray(q, dtype=np.float64)
        single = q_arr.ndim == 1
        if single:
            q_arr = q_arr[None, :]
        flat = self.mean + q_arr @ self.components
        out = flat.reshape(len(q_arr), -1, 3)
        return out[0] if single else out

    @staticmethod
    def decode_torch(
        q: torch.Tensor,
        mean_t: torch.Tensor,
        basis_t: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable PyTorch reconstruction of shape (B, n_elem, 3)."""
        flat = mean_t + q @ basis_t
        return flat.view(q.shape[0], -1, 3)

    def to_state_dict(self) -> Dict[str, np.ndarray]:
        assert self.mean is not None and self.components is not None
        return {
            "mean": self.mean,
            "components": self.components,
            "singular_values": self.singular_values if self.singular_values is not None else np.array([]),
            "captured_energy": np.array([self.captured_energy], dtype=np.float64),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, np.ndarray]) -> "PODBasis":
        obj = cls()
        obj.mean = np.asarray(state["mean"], dtype=np.float64)
        obj.components = np.asarray(state["components"], dtype=np.float64)
        obj.singular_values = np.asarray(state.get("singular_values", []), dtype=np.float64)
        obj.captured_energy = float(state["captured_energy"][0])
        return obj


def fit_intact_compatibility_matrix(
    strains_norm_tr: np.ndarray,
    labels_tr: np.ndarray,
    n_sensors: int = 12,
    n_ch: int = 3,
    intact_label: int = 0,
) -> np.ndarray:
    """Fit block-zero-diagonal leave-one-sensor-out intact strain compatibility matrix."""
    s_flat = np.asarray(strains_norm_tr, dtype=np.float64).reshape(len(strains_norm_tr), -1)
    intact_mask = labels_tr == intact_label
    s_intact = s_flat[intact_mask] if np.sum(intact_mask) >= 6 else s_flat
    dim = n_sensors * n_ch
    w_compat = np.zeros((dim, dim), dtype=np.float64)
    alphas = np.logspace(-4, 1, 15)
    for sens in range(n_sensors):
        cols = [n_ch * sens + c for c in range(n_ch)]
        other = [c for c in range(dim) if c not in cols]
        reg = RidgeCV(alphas=alphas).fit(s_intact[:, other], s_intact[:, cols])
        w_compat[np.ix_(cols, other)] = reg.coef_
    return w_compat


def fit_strain_covariance(strains_norm_tr: np.ndarray, reg: float = 2e-3) -> np.ndarray:
    """Fit regularized (36, 36) strain covariance for minimum-variance conditional imputation."""
    s_flat = np.asarray(strains_norm_tr, dtype=np.float64).reshape(len(strains_norm_tr), -1)
    cov = np.cov(s_flat, rowvar=False)
    return cov + float(reg) * np.eye(s_flat.shape[1], dtype=np.float64)


def fit_elastic_pod_linear(
    strains_norm_tr: np.ndarray,
    q_tr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit linear elastic baseline mapping standardized strains (N, 36) -> POD coords (N, k)."""
    s_flat = np.asarray(strains_norm_tr, dtype=np.float64).reshape(len(strains_norm_tr), -1)
    reg = RidgeCV(alphas=np.logspace(-4, 2, 20)).fit(s_flat, q_tr)
    return np.asarray(reg.coef_, dtype=np.float64), np.asarray(reg.intercept_, dtype=np.float64)


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block with SiLU activation and dropout."""

    def __init__(self, dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(F.silu(self.norm1(x)))
        h = self.drop(h)
        h = self.fc2(F.silu(self.norm2(h)))
        return x + h


class AeroRFShapeNet(nn.Module):
    """Decoupled Physics-Informed shape-sensing & battle-damage network.

    Combines exact Conditional Minimum-Variance Elastic Imputation for dead FBG sensors
    with a linear-elastic POD baseline + residual shape branch and a load-invariant
    parity-space strain-compatibility residual branch for battle-damage classification.
    """

    def __init__(
        self,
        n_sensors: int = 12,
        n_channels_per_sensor: int = 3,
        n_modes: int = 12,
        n_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.15,
        sigma_strain: Optional[np.ndarray] = None,
        w_compat: Optional[np.ndarray] = None,
        w_lin_pod: Optional[np.ndarray] = None,
        b_lin_pod: Optional[np.ndarray] = None,
        lambda_obs: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.n_channels_per_sensor = n_channels_per_sensor
        self.n_modes = n_modes
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lambda_obs = float(lambda_obs)

        strain_dim = n_sensors * n_channels_per_sensor
        feat_dim = strain_dim * 2 + n_sensors  # 36 + 36 + 12 = 84

        # Frozen structural-physics buffers (saved inside model state_dict)
        sig_init = (
            np.eye(strain_dim, dtype=np.float32)
            if sigma_strain is None
            else np.asarray(sigma_strain, dtype=np.float32)
        )
        wcom_init = (
            np.zeros((strain_dim, strain_dim), dtype=np.float32)
            if w_compat is None
            else np.asarray(w_compat, dtype=np.float32)
        )
        wlin_init = (
            np.zeros((n_modes, strain_dim), dtype=np.float32)
            if w_lin_pod is None
            else np.asarray(w_lin_pod, dtype=np.float32)
        )
        blin_init = (
            np.zeros((n_modes,), dtype=np.float32)
            if b_lin_pod is None
            else np.asarray(b_lin_pod, dtype=np.float32)
        )

        self.register_buffer("sigma_strain", torch.tensor(sig_init, dtype=torch.float32))
        self.register_buffer("w_compat", torch.tensor(wcom_init, dtype=torch.float32))
        self.register_buffer("w_lin_pod", torch.tensor(wlin_init, dtype=torch.float32))
        self.register_buffer("b_lin_pod", torch.tensor(blin_init, dtype=torch.float32))

        # Decoupled Shape Residual Branch: [s_imp (36), r_norm (36), mask (12)] -> delta_q
        self.shape_in = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.shape_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout=dropout),
            ResidualBlock(hidden_dim, dropout=dropout),
        )
        self.shape_out_norm = nn.LayerNorm(hidden_dim)
        self.shape_head = nn.Linear(hidden_dim, n_modes)
        nn.init.normal_(self.shape_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.shape_head.bias)

        # Decoupled Damage Parity Branch: [r_norm (36), |r_norm| (36), mask (12)] -> logits, sev
        self.damage_in = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.damage_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout=dropout),
            ResidualBlock(hidden_dim, dropout=dropout),
        )
        self.damage_out_norm = nn.LayerNorm(hidden_dim)
        self.damage_head = nn.Linear(hidden_dim, n_classes)
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def shape_parameters(self):
        for mod in (self.shape_in, self.shape_blocks, self.shape_out_norm, self.shape_head):
            yield from mod.parameters()

    def damage_parameters(self):
        for mod in (
            self.damage_in,
            self.damage_blocks,
            self.damage_out_norm,
            self.damage_head,
            self.severity_head,
        ):
            yield from mod.parameters()

    def impute_masked(self, s36: torch.Tensor, mask12: torch.Tensor) -> torch.Tensor:
        """Exact conditional Gaussian minimum-variance imputation for severed FBG sensors."""
        out = s36.clone()
        m36 = mask12.repeat_interleave(self.n_channels_per_sensor, dim=1) > 0.5
        if torch.all(m36):
            return out

        eye = torch.eye(s36.shape[1], device=s36.device, dtype=s36.dtype)
        for i in range(s36.shape[0]):
            obs = m36[i]
            if torch.all(obs):
                continue
            obs_idx = torch.nonzero(obs, as_tuple=True)[0]
            mis_idx = torch.nonzero(~obs, as_tuple=True)[0]
            if len(obs_idx) == 0:
                continue
            n_o = len(obs_idx)
            s_oo = self.sigma_strain[obs_idx][:, obs_idx] + self.lambda_obs * eye[:n_o, :n_o]
            s_mo = self.sigma_strain[mis_idx][:, obs_idx]
            w_obs = torch.linalg.solve(s_oo, s36[i, obs_idx].unsqueeze(-1)).squeeze(-1)
            out[i, mis_idx] = s_mo @ w_obs
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        strain_dim = self.n_sensors * self.n_channels_per_sensor
        s36 = x[:, :strain_dim]
        mask12 = x[:, strain_dim:]

        s_imp = self.impute_masked(s36, mask12)
        m36 = mask12.repeat_interleave(self.n_channels_per_sensor, dim=1)

        # Load-invariant parity-space compatibility residual on observed healthy channels
        r_raw = (s_imp - s_imp @ self.w_compat.T) * m36
        s_norm = torch.norm(s_imp * m36, dim=1, keepdim=True) + 0.1
        r_norm = r_raw / s_norm

        # 1. Shape branch: elastic linear POD baseline + nonlinear correction
        q_base = s_imp @ self.w_lin_pod.T + self.b_lin_pod
        x_shape = torch.cat([s_imp, r_norm, mask12], dim=1)
        hs = self.shape_out_norm(self.shape_blocks(self.shape_in(x_shape)))
        modal_coeffs = q_base + self.shape_head(hs)

        # 2. Damage branch: decoupled parity-space topology & severity estimator
        x_dmg = torch.cat([r_norm, torch.abs(r_norm), mask12], dim=1)
        hd = self.damage_out_norm(self.damage_blocks(self.damage_in(x_dmg)))
        damage_logits = self.damage_head(hd)
        severity = self.severity_head(hd)

        return modal_coeffs, damage_logits, severity


class RadarPhasePhysicsLoss(nn.Module):
    """Joint structural displacement MSE + X-Band RF phase coherence + damage topology loss."""

    def __init__(
        self,
        k0: float,
        mean_vec: np.ndarray,
        basis_mat: np.ndarray,
        steer_angles_deg: Tuple[float, ...] = (0.0, 15.0, 30.0),
        w_rf: float = 15.0,
        w_cls: float = 1.0,
        w_sev: float = 0.5,
    ) -> None:
        super().__init__()
        self.k0 = float(k0)
        self.w_rf = float(w_rf)
        self.w_cls = float(w_cls)
        self.w_sev = float(w_sev)

        u0_list = [
            [math.sin(math.radians(a)), 0.0, math.cos(math.radians(a))]
            for a in steer_angles_deg
        ]
        self.register_buffer("u0_dirs", torch.tensor(u0_list, dtype=torch.float32))
        self.register_buffer("mean_t", torch.tensor(mean_vec, dtype=torch.float32))
        self.register_buffer("basis_t", torch.tensor(basis_mat, dtype=torch.float32))

    def shape_loss(
        self,
        modal_pred: torch.Tensor,
        disp_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        disp_pred = PODBasis.decode_torch(modal_pred, self.mean_t, self.basis_t)
        disp_mse_mm2 = F.mse_loss(disp_pred * 1.0e3, disp_true * 1.0e3)
        err_m = disp_pred - disp_true
        phase_err = self.k0 * torch.einsum("bni,mi->bnm", err_m, self.u0_dirs)
        rf_phase_loss = torch.mean(1.0 - torch.cos(phase_err))
        loss = disp_mse_mm2 + self.w_rf * rf_phase_loss
        return loss, float(disp_mse_mm2.item()), float(rf_phase_loss.item())

    def damage_loss(
        self,
        damage_logits: torch.Tensor,
        sev_pred: torch.Tensor,
        label_true: torch.Tensor,
        sev_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        cls_loss = F.cross_entropy(damage_logits, label_true)
        sev_loss = F.mse_loss(sev_pred.squeeze(-1), sev_true)
        loss = self.w_cls * cls_loss + self.w_sev * sev_loss
        return loss, float(cls_loss.item()), float(sev_loss.item())

    def forward(
        self,
        modal_pred: torch.Tensor,
        damage_logits: torch.Tensor,
        sev_pred: torch.Tensor,
        disp_true: torch.Tensor,
        label_true: torch.Tensor,
        sev_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_shape, mse_mm2, rf_ph = self.shape_loss(modal_pred, disp_true)
        l_dmg, cls_l, sev_l = self.damage_loss(damage_logits, sev_pred, label_true, sev_true)
        total = l_shape + l_dmg
        parts = {
            "total": float(total.item()),
            "disp_mse_mm2": mse_mm2,
            "rf_phase": rf_ph,
            "cls": cls_l,
            "sev": sev_l,
        }
        return total, parts


def apply_combat_sensor_augmentation(
    strains_norm: np.ndarray,
    noise_std: float = 0.03,
    max_dropouts: int = 2,
    exact_dropouts: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply sensor noise + random severed FBG dropouts to standardized strains (N, 12, 3)."""
    if rng is None:
        rng = np.random.default_rng()
    strains_norm = np.asarray(strains_norm, dtype=np.float64)
    n, n_sens, n_ch = strains_norm.shape

    noisy = strains_norm.copy()
    if noise_std > 0.0:
        noisy += rng.normal(0.0, noise_std, size=noisy.shape)

    mask = np.ones((n, n_sens), dtype=np.float64)
    if max_dropouts > 0:
        for i in range(n):
            k_drop = max_dropouts if exact_dropouts else int(rng.integers(0, max_dropouts + 1))
            if k_drop > 0:
                dead_idx = rng.choice(n_sens, size=k_drop, replace=False)
                mask[i, dead_idx] = 0.0
                noisy[i, dead_idx, :] = 0.0

    strains_flat_36 = noisy.reshape(n, n_sens * n_ch)
    x_48 = np.concatenate([strains_flat_36, mask], axis=1)
    return x_48, strains_flat_36, mask


class RidgeBaseline:
    """Linear RidgeCV baseline mapping 36 FBG strain channels to 768 displacements."""

    def __init__(self, alphas: Optional[np.ndarray] = None) -> None:
        self.scaler = StandardScaler()
        alpha_grid = np.logspace(-3, 4, 29) if alphas is None else alphas
        self.model = RidgeCV(alphas=alpha_grid)

    def fit(self, fbg_strains: np.ndarray, disp_xyz: np.ndarray) -> "RidgeBaseline":
        x_36 = np.asarray(fbg_strains, dtype=np.float64).reshape(len(fbg_strains), -1)
        y_768 = np.asarray(disp_xyz, dtype=np.float64).reshape(len(disp_xyz), -1)
        x_scaled = self.scaler.fit_transform(x_36)
        self.model.fit(x_scaled, y_768)
        return self

    def predict(self, strains_flat_36_norm: np.ndarray) -> np.ndarray:
        """Predict (N, 256, 3) displacements from already-standardized (N, 36) FBG inputs."""
        y_hat = self.model.predict(strains_flat_36_norm)
        return y_hat.reshape(len(strains_flat_36_norm), -1, 3)


@dataclass
class TrainHistory:
    epochs: int
    best_epoch: int
    best_val_loss: float
    device_used: str
    train_seconds: float


def _clone_params(params) -> list[torch.Tensor]:
    return [p.detach().cpu().clone() for p in params]


def _load_params(params, saved: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, s in zip(params, saved):
            p.copy_(s.to(p.device))


def train_shapenet(
    fbg_train: np.ndarray,
    disp_train: np.ndarray,
    label_train: np.ndarray,
    sev_train: np.ndarray,
    array_cfg: ArrayConfig,
    max_modes: int = 12,
    hidden_dim: int = 128,
    epochs: int = 350,
    batch_size: int = 16,
    lr: float = 1.5e-3,
    seed: int = 42,
) -> Tuple[AeroRFShapeNet, PODBasis, np.ndarray, np.ndarray, TrainHistory]:
    """Train decoupled AeroRFShapeNet with physics operators fitted solely on tr_idx."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = select_torch_device()

    strain_mean = fbg_train.mean(axis=0, keepdims=True)
    strain_std = np.maximum(fbg_train.std(axis=0, keepdims=True), 1e-8)
    strains_norm = (fbg_train - strain_mean) / strain_std

    # Stratified internal train/val split (85/15) so test set stays 100% unseen
    tr_idx_list, va_idx_list = [], []
    for cls_id in np.unique(label_train):
        idx_c = np.where(label_train == cls_id)[0]
        rng.shuffle(idx_c)
        n_va = max(1, int(round(0.15 * len(idx_c))))
        va_idx_list.append(idx_c[:n_va])
        tr_idx_list.append(idx_c[n_va:])
    tr_idx = np.concatenate(tr_idx_list)
    va_idx = np.concatenate(va_idx_list)

    # Fit POD + physics operators ONLY on tr_idx
    pod = PODBasis().fit(disp_train[tr_idx], max_modes=max_modes)
    assert pod.mean is not None and pod.components is not None
    q_tr = pod.encode(disp_train[tr_idx])

    sigma_strain = fit_strain_covariance(strains_norm[tr_idx], reg=2e-3)
    w_compat = fit_intact_compatibility_matrix(
        strains_norm[tr_idx],
        label_train[tr_idx],
        n_sensors=fbg_train.shape[1],
        n_ch=fbg_train.shape[2],
    )
    w_lin_pod, b_lin_pod = fit_elastic_pod_linear(strains_norm[tr_idx], q_tr)

    model = AeroRFShapeNet(
        n_sensors=fbg_train.shape[1],
        n_channels_per_sensor=fbg_train.shape[2],
        n_modes=pod.n_modes,
        n_classes=len(DAMAGE_CLASSES),
        hidden_dim=hidden_dim,
        dropout=0.15,
        sigma_strain=sigma_strain,
        w_compat=w_compat,
        w_lin_pod=w_lin_pod,
        b_lin_pod=b_lin_pod,
    ).to(device)

    criterion = RadarPhasePhysicsLoss(
        k0=array_cfg.k0,
        mean_vec=pod.mean,
        basis_mat=pod.components,
    ).to(device)

    opt_shape = torch.optim.AdamW(model.shape_parameters(), lr=lr, weight_decay=3e-4)
    opt_damage = torch.optim.AdamW(model.damage_parameters(), lr=lr, weight_decay=3e-4)
    sched_shape = torch.optim.lr_scheduler.CosineAnnealingLR(opt_shape, T_max=epochs, eta_min=5e-5)
    sched_damage = torch.optim.lr_scheduler.CosineAnnealingLR(opt_damage, T_max=epochs, eta_min=5e-5)

    # Multi-regime validation pack (Clean + 3% Noise + 1-Dead + 2-Dead) for stable checkpointing
    val_rng = np.random.default_rng(seed + 999)
    val_x_list, val_d_list, val_l_list, val_s_list = [], [], [], []
    for n_std, k_drop in ((0.0, 0), (0.03, 0), (0.03, 1), (0.03, 2)):
        xv, _, _ = apply_combat_sensor_augmentation(
            strains_norm[va_idx],
            noise_std=n_std,
            max_dropouts=k_drop,
            exact_dropouts=True,
            rng=val_rng,
        )
        val_x_list.append(xv)
        val_d_list.append(disp_train[va_idx])
        val_l_list.append(label_train[va_idx])
        val_s_list.append(sev_train[va_idx])

    x_va_t = torch.tensor(np.vstack(val_x_list), dtype=torch.float32, device=device)
    y_va_disp_t = torch.tensor(np.vstack(val_d_list), dtype=torch.float32, device=device)
    y_va_lbl_t = torch.tensor(np.concatenate(val_l_list), dtype=torch.long, device=device)
    y_va_sev_t = torch.tensor(np.concatenate(val_s_list), dtype=torch.float32, device=device)

    best_shape_loss = float("inf")
    best_dmg_loss = float("inf")
    best_shape_epoch = 0
    best_dmg_epoch = 0
    best_shape_params = _clone_params(model.shape_parameters())
    best_dmg_params = _clone_params(model.damage_parameters())

    t0 = time.perf_counter()
    n_tr = len(tr_idx)
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(tr_idx)
        for start in range(0, n_tr, batch_size):
            b_idx = perm[start : start + batch_size]
            noise_lvl = float(rng.uniform(0.0, 0.04))
            x_b_48, _, _ = apply_combat_sensor_augmentation(
                strains_norm[b_idx],
                noise_std=noise_lvl,
                max_dropouts=2,
                exact_dropouts=False,
                rng=rng,
            )
            xb = torch.tensor(x_b_48, dtype=torch.float32, device=device)
            yd = torch.tensor(disp_train[b_idx], dtype=torch.float32, device=device)
            yl = torch.tensor(label_train[b_idx], dtype=torch.long, device=device)
            ys = torch.tensor(sev_train[b_idx], dtype=torch.float32, device=device)

            opt_shape.zero_grad(set_to_none=True)
            opt_damage.zero_grad(set_to_none=True)

            q_hat, logits, sev_hat = model(xb)
            l_shape, _, _ = criterion.shape_loss(q_hat, yd)
            l_dmg, _, _ = criterion.damage_loss(logits, sev_hat, yl, ys)
            (l_shape + l_dmg).backward()

            torch.nn.utils.clip_grad_norm_(model.shape_parameters(), 2.0)
            torch.nn.utils.clip_grad_norm_(model.damage_parameters(), 2.0)
            opt_shape.step()
            opt_damage.step()

        sched_shape.step()
        sched_damage.step()

        model.eval()
        with torch.no_grad():
            vq, vlogits, vsev = model(x_va_t)
            vs_loss, _, _ = criterion.shape_loss(vq, y_va_disp_t)
            vd_loss, _, _ = criterion.damage_loss(vlogits, vsev, y_va_lbl_t, y_va_sev_t)
            vs_val = float(vs_loss.item())
            vd_val = float(vd_loss.item())

        if vs_val < best_shape_loss:
            best_shape_loss = vs_val
            best_shape_epoch = ep
            best_shape_params = _clone_params(model.shape_parameters())
        if vd_val < best_dmg_loss:
            best_dmg_loss = vd_val
            best_dmg_epoch = ep
            best_dmg_params = _clone_params(model.damage_parameters())

    train_seconds = time.perf_counter() - t0
    _load_params(model.shape_parameters(), best_shape_params)
    _load_params(model.damage_parameters(), best_dmg_params)
    model.eval()

    hist = TrainHistory(
        epochs=epochs,
        best_epoch=max(best_shape_epoch, best_dmg_epoch),
        best_val_loss=best_shape_loss + best_dmg_loss,
        device_used=str(device),
        train_seconds=train_seconds,
    )
    return model, pod, strain_mean, strain_std, hist


def predict_with_latency(
    model: AeroRFShapeNet,
    pod: PODBasis,
    x_48: np.ndarray,
    device: Optional[torch.device] = None,
    n_warmup: int = 10,
    n_bench: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run inference and measure true single-sample inference latency (ms)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    x_t = torch.tensor(x_48, dtype=torch.float32, device=device)
    with torch.no_grad():
        q_hat, logits, sev_hat = model(x_t)

    disp_pred = pod.decode_numpy(q_hat.cpu().numpy())
    cls_pred = np.argmax(logits.cpu().numpy(), axis=1)
    sev_pred = sev_hat.squeeze(-1).cpu().numpy()

    single_t = x_t[:1]
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(single_t)
        t0 = time.perf_counter()
        for _ in range(n_bench):
            q1, _, _ = model(single_t)
            _ = pod.decode_numpy(q1[0].cpu().numpy())
        latency_ms = 1e3 * (time.perf_counter() - t0) / n_bench

    return disp_pred, cls_pred, sev_pred, float(latency_ms)


def save_checkpoint(
    ckpt_path: Path,
    model: AeroRFShapeNet,
    pod: PODBasis,
    strain_mean: np.ndarray,
    strain_std: np.ndarray,
) -> Path:
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": {
                "n_sensors": model.n_sensors,
                "n_channels_per_sensor": model.n_channels_per_sensor,
                "n_modes": model.n_modes,
                "n_classes": model.n_classes,
                "hidden_dim": model.hidden_dim,
                "dropout": model.dropout,
                "lambda_obs": model.lambda_obs,
            },
            "pod_state": pod.to_state_dict(),
            "strain_mean": strain_mean,
            "strain_std": strain_std,
        },
        ckpt_path,
    )
    return ckpt_path


def load_checkpoint(
    ckpt_path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[AeroRFShapeNet, PODBasis, np.ndarray, np.ndarray]:
    if device is None:
        device = select_torch_device()
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AeroRFShapeNet(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    pod = PODBasis.from_state_dict(payload["pod_state"])
    return model, pod, payload["strain_mean"], payload["strain_std"]

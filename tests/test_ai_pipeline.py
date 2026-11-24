"""Unit tests for Phase 2 Physics-Informed POD ShapeNet AI pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerorf.config import ArrayConfig
from aerorf.shape_ai import (
    AeroRFShapeNet,
    PODBasis,
    RadarPhasePhysicsLoss,
    apply_combat_sensor_augmentation,
    fit_strain_covariance,
    predict_with_latency,
)


class TestAIPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ArrayConfig()
        self.rng = np.random.default_rng(7)
        pos = self.cfg.nominal_element_positions()
        x_rel = (pos[:, 0] - self.cfg.x_center) / 0.12
        y_rel = (pos[:, 1] - self.cfg.y_center) / 0.12

        # Synthetic 4-mode elastic displacement field (N=40, 256, 3)
        n = 40
        c = self.rng.normal(0.0, 0.01, size=(n, 4))
        basis = np.stack(
            [
                np.column_stack([np.zeros(256), np.zeros(256), x_rel]),
                np.column_stack([np.zeros(256), np.zeros(256), x_rel**2]),
                np.column_stack([np.zeros(256), np.zeros(256), x_rel * y_rel]),
                np.column_stack([0.1 * x_rel, 0.05 * y_rel, y_rel**2]),
            ],
            axis=0,
        )
        self.disp = np.einsum("nm,mij->nij", c, basis)

    def test_pod_basis_energy_and_roundtrip(self) -> None:
        """POD captures >99.9% energy and reconstructs multi-mode deflections accurately."""
        pod = PODBasis().fit(self.disp, max_modes=8)
        self.assertGreater(pod.energy_captured(), 0.999)
        q = pod.encode(self.disp)
        rec = pod.decode_numpy(q)
        max_err_mm = np.max(np.abs(rec - self.disp)) * 1e3
        self.assertLess(max_err_mm, 1e-6)

    def test_shapenet_forward_mask_and_physics_loss(self) -> None:
        """AeroRFShapeNet handles masked dropouts, flows gradients, and runs <5ms."""
        pod = PODBasis().fit(self.disp, max_modes=8)
        assert pod.mean is not None and pod.components is not None
        model = AeroRFShapeNet(n_modes=pod.n_modes, hidden_dim=96)
        criterion = RadarPhasePhysicsLoss(
            k0=self.cfg.k0,
            mean_vec=pod.mean,
            basis_mat=pod.components,
        )

        strains = self.rng.normal(0.0, 1.0, size=(8, 12, 3))
        x_48, s36, mask = apply_combat_sensor_augmentation(
            strains, noise_std=0.03, max_dropouts=2, exact_dropouts=True, rng=self.rng
        )
        self.assertEqual(x_48.shape, (8, 48))
        self.assertEqual(s36.shape, (8, 36))
        self.assertTrue(np.all(mask.sum(axis=1) == 10.0))

        xb = torch.tensor(x_48, dtype=torch.float32)
        yd = torch.tensor(self.disp[:8], dtype=torch.float32)
        yl = torch.zeros(8, dtype=torch.long)
        ys = torch.ones(8, dtype=torch.float32)

        q_hat, logits, sev = model(xb)
        self.assertEqual(q_hat.shape, (8, pod.n_modes))
        self.assertEqual(logits.shape, (8, 4))
        self.assertEqual(sev.shape, (8, 1))

        loss, parts = criterion(q_hat, logits, sev, yd, yl, ys)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("rf_phase", parts)
        loss.backward()
        grad_norm = sum(
            p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

        disp_pred, cls_pred, sev_pred, latency_ms = predict_with_latency(
            model, pod, x_48, n_warmup=5, n_bench=50
        )
        self.assertEqual(disp_pred.shape, (8, 256, 3))
        self.assertEqual(cls_pred.shape, (8,))
        self.assertEqual(sev_pred.shape, (8,))
        self.assertLess(latency_ms, 5.0)

    def test_conditional_elastic_imputation_recovers_dead_sensors(self) -> None:
        """Conditional minimum-variance imputation accurately recovers severed sensor channels."""
        # Correlated 12-sensor synthetic strain field driven by 3 latent load factors
        z_lat = self.rng.normal(0.0, 1.0, size=(60, 3))
        proj = self.rng.normal(0.0, 1.0, size=(3, 36))
        s_clean = z_lat @ proj
        s_clean = (s_clean - s_clean.mean(0, keepdims=True)) / np.maximum(s_clean.std(0, keepdims=True), 1e-6)
        sigma = fit_strain_covariance(s_clean[:45], reg=1e-4)

        model = AeroRFShapeNet(n_modes=8, hidden_dim=64, sigma_strain=sigma, lambda_obs=1e-4)
        mask = np.ones((5, 12), dtype=np.float64)
        mask[:, [2, 7]] = 0.0
        s_masked = s_clean[45:50].copy()
        s_masked[:, [6, 7, 8, 21, 22, 23]] = 0.0

        s_imp = model.impute_masked(
            torch.tensor(s_masked, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        ).numpy()

        dropped_err = np.sqrt(np.mean((s_imp[:, [6, 7, 8, 21, 22, 23]] - s_clean[45:50][:, [6, 7, 8, 21, 22, 23]]) ** 2))
        self.assertLess(dropped_err, 0.05)


if __name__ == "__main__":
    unittest.main()

"""Verification & validation suite for the IEEE-exact X-band conformal RF engine."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerorf.config import ArrayConfig
from aerorf.rf_array import (
    compute_pattern_cut,
    compute_steering_weights,
    evaluate_conformal_scenario,
    evaluate_pattern_metrics,
)


class TestRFValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ArrayConfig()
        self.theta = np.linspace(-90.0, 90.0, 3601)

    def test_uniform_broadside_sidelobe_level(self) -> None:
        """Verify 16x16 uniform broadside array matches theoretical -13.15 dB +/- 0.20 dB SLL."""
        pos = self.cfg.nominal_element_positions()
        norm = self.cfg.nominal_normals()
        amp_uniform = np.full(self.cfg.num_elements, 1.0 / self.cfg.num_elements)
        weights = compute_steering_weights(
            pos, theta0_deg=0.0, phi0_deg=0.0, k=self.cfg.k0, amp_weights=amp_uniform
        )
        pat_db, _ = compute_pattern_cut(
            pos, norm, weights, self.cfg.k0, self.theta, phi_deg=0.0, q_elem=self.cfg.q_elem
        )
        metrics = evaluate_pattern_metrics(self.theta, pat_db, target_theta_deg=0.0)
        self.assertAlmostEqual(metrics["sll_db"], -13.15, delta=0.20)
        self.assertLess(abs(metrics["boresight_error_deg"]), 0.05)

    def test_taylor_30db_undeformed_broadside(self) -> None:
        """Verify Taylor -30 dB tapered undeformed array meets SLL <= -29 dB and zero boresight error."""
        zeros_disp = np.zeros((self.cfg.num_elements, 3), dtype=np.float64)
        zeros_rot = np.zeros((self.cfg.num_elements, 3), dtype=np.float64)
        res = evaluate_conformal_scenario(
            self.cfg, zeros_disp, zeros_rot, theta0_deg=0.0, phi0_deg=0.0
        )
        nom = res["metrics_primary"]["nominal"]
        self.assertLessEqual(nom["sll_db"], -29.0)
        self.assertLess(abs(nom["boresight_error_deg"]), 0.05)

    def test_rigid_wing_tilt_3deg_self_healing(self) -> None:
        """Verify 3.0 deg rigid spanwise tilt shifts uncompensated beam by ~3.0 deg and 6-bit healing restores it."""
        pos = self.cfg.nominal_element_positions()
        tilt_deg = 3.0
        tilt_rad = math.radians(tilt_deg)

        disp = np.zeros_like(pos)
        disp[:, 2] = (pos[:, 0] - self.cfg.x_center) * math.tan(tilt_rad)

        # Upward spanwise slope (+dz/dx) tilts normal toward -x -> negative roty
        rot = np.zeros_like(pos)
        rot[:, 1] = -tilt_rad

        res = evaluate_conformal_scenario(
            self.cfg, disp, rot, theta0_deg=0.0, phi0_deg=0.0, quant_bits=6
        )
        uncomp = res["metrics_az"]["deformed_uncompensated"]
        healed = res["metrics_az"]["healed_quantized"]

        self.assertAlmostEqual(abs(uncomp["boresight_error_deg"]), 3.0, delta=0.15)
        self.assertLess(abs(healed["boresight_error_deg"]), 0.10)
        self.assertLess(healed["gain_loss_db"], 0.25)

    def test_severe_quadratic_bending_self_healing(self) -> None:
        """Verify severe quadratic wing bending (~240 deg phase error) is healed by 6-bit T/R compensation."""
        pos = self.cfg.nominal_element_positions()
        half_span = 0.5 * (self.cfg.nx - 1) * self.cfg.dx
        dx_rel = pos[:, 0] - self.cfg.x_center

        # 240 deg phase error across aperture (~20.0 mm edge sagitta at 10 GHz)
        w_edge = (240.0 / 360.0) * self.cfg.lambda0
        disp = np.zeros_like(pos)
        disp[:, 2] = w_edge * (dx_rel / half_span) ** 2

        slope = 2.0 * w_edge * dx_rel / (half_span**2)
        rot = np.zeros_like(pos)
        rot[:, 1] = -np.arctan(slope)

        res = evaluate_conformal_scenario(
            self.cfg, disp, rot, theta0_deg=0.0, phi0_deg=0.0, quant_bits=6
        )
        uncomp = res["metrics_az"]["deformed_uncompensated"]
        healed_q = res["metrics_az"]["healed_quantized"]
        healed_id = res["metrics_az"]["healed_ideal"]

        self.assertGreater(uncomp["gain_loss_db"], 3.0)
        self.assertGreater(uncomp["sll_db"], -15.0)

        self.assertLess(healed_id["gain_loss_db"], 0.30)
        self.assertLess(healed_q["gain_loss_db"], 0.30)
        self.assertLess(healed_id["sll_db"], -25.0)
        self.assertLess(healed_q["sll_db"], -25.0)


if __name__ == "__main__":
    unittest.main()

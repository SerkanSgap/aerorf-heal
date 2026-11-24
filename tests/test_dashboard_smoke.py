"""Smoke tests for Phase 3 Streamlit cockpit backend and Plotly figure builders."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerorf.dashboard_core import (
    load_cockpit_assets,
    make_1d_pattern_figure,
    make_2d_uv_comparison_figure,
    make_3d_aperture_figure,
    make_fbg_parity_figure,
    run_live_combat_inference,
)


class TestDashboardSmoke(unittest.TestCase):
    def test_cockpit_assets_inference_and_figures(self) -> None:
        """Load assets, run 2-dead-sensor live inference, and construct all 4 Plotly figures."""
        assets = load_cockpit_assets(ROOT)
        self.assertEqual(len(assets.damage_label), 120)
        self.assertEqual(int(assets.heldout_mask.sum()), 24)

        heldout_damaged = [
            int(i) for i in np.where(assets.heldout_mask)[0] if assets.damage_label[i] != 0
        ]
        self.assertGreater(len(heldout_damaged), 0)
        case_idx = heldout_damaged[0]

        res = run_live_combat_inference(
            assets=assets,
            case_idx=case_idx,
            dead_sensor_ids=[3, 7],
            noise_pct=3.0,
            theta0_deg=15.0,
            quant_bits=6,
            rng_seed=777,
        )
        self.assertTrue(res["is_heldout"])
        self.assertEqual(res["disp_ai"].shape, (256, 3))
        self.assertEqual(res["sensor_parity_scores"].shape, (12,))
        self.assertLess(res["latency_ms"], 10.0)
        self.assertTrue(np.all(np.isfinite(res["uv_healed"])))

        figs = [
            make_3d_aperture_figure(
                assets.wing_cfg,
                assets.array_cfg,
                assets.fbg_cfg,
                res["disp_true"],
                res["disp_ai"],
                [3, 7],
                res["pred_class"],
            ),
            make_1d_pattern_figure(res),
            make_2d_uv_comparison_figure(res),
            make_fbg_parity_figure(res),
        ]
        for fig in figs:
            self.assertIsInstance(fig, go.Figure)
            self.assertGreater(len(fig.data), 0)

    def test_showcase_assets_present_and_valid(self) -> None:
        """Verify generated LinkedIn showcase PNGs, GIF, and PDF carousel are valid."""
        from PIL import Image
        from aerorf.showcase_renderer import pick_dramatic_heldout_case

        assets = load_cockpit_assets(ROOT)
        best_idx = pick_dramatic_heldout_case(assets)
        self.assertTrue(assets.heldout_mask[best_idx])
        self.assertNotEqual(int(assets.damage_label[best_idx]), 0)

        showcase_dir = ROOT / "outputs" / "showcase"
        for name in (
            "slide1_story_problem.png",
            "slide2_combat_sensor_kill.png",
            "slide3_architecture_and_vv.png",
        ):
            p = showcase_dir / name
            self.assertTrue(p.is_file(), f"Missing {p}")
            with Image.open(p) as im:
                self.assertEqual(im.size, (1600, 900))

        gif_path = showcase_dir / "aerorf_healing_demo.gif"
        self.assertTrue(gif_path.is_file(), f"Missing {gif_path}")
        with Image.open(gif_path) as gim:
            self.assertEqual(gim.size, (1080, 620))
            self.assertGreaterEqual(getattr(gim, "n_frames", 1), 24)

        pdf_path = showcase_dir / "AeroRF_Heal_LinkedIn_Carousel.pdf"
        self.assertTrue(pdf_path.is_file(), f"Missing {pdf_path}")
        self.assertGreater(pdf_path.stat().st_size, 50_000)


if __name__ == "__main__":
    unittest.main()

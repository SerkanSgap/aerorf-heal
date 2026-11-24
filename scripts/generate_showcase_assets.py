"""Generate ready-to-upload LinkedIn storytelling visuals and animated GIF for AeroRF-Heal."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerorf.showcase_renderer import generate_all_showcase_assets


def main() -> None:
    info = generate_all_showcase_assets(ROOT)
    print(f"Held-out ANSYS Case #{info['case_idx']} ({info['true_class']} -> AI predicted: {info['pred_class']})")
    print(f"Uncompensated: {info['uncomp_err_deg']:+.2f} deg | SLL {info['uncomp_sll_db']:.2f} dB")
    print(f"AI Self-Healed: {info['ai_err_deg']:+.2f} deg | SLL {info['ai_sll_db']:.2f} dB")
    print(f"GIF frames: {info['gif_frames']}")
    for p in info["files"]:
        kb = p.stat().st_size / 1024.0
        print(f"Saved: {p} ({kb:.1f} KB)")


if __name__ == "__main__":
    main()

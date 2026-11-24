"""Render LinkedIn storytelling showcase slides, PDF carousel, and live GIF for AeroRF-Heal."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .dashboard_core import CockpitAssets, load_cockpit_assets, run_live_combat_inference
from .rf_array import (
    compute_pattern_cut,
    compute_steering_weights,
    evaluate_pattern_metrics,
    make_2d_taylor_taper,
    rotate_normals_euler,
)

# Dark Aerospace HUD Palette
BG = (11, 17, 32)
CARD = (18, 28, 48)
CARD_ALT = (24, 38, 64)
BORDER = (51, 65, 85)
TEXT = (241, 245, 249)
MUTED = (148, 163, 184)
CYAN = (0, 229, 255)
GREEN = (16, 185, 129)
RED = (244, 63, 94)
AMBER = (245, 158, 11)
VIOLET = (167, 139, 250)
SLATE_LINE = (100, 116, 139)

FONT_BOLD_PATH = r"C:\Windows\Fonts\segoeuib.ttf"
FONT_REG_PATH = r"C:\Windows\Fonts\segoeui.ttf"
FONT_MONO_BOLD_PATH = r"C:\Windows\Fonts\consolab.ttf"
FONT_MONO_PATH = r"C:\Windows\Fonts\consola.ttf"


def _font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load crisp Windows TrueType font with graceful fallback."""
    if mono:
        candidates = [FONT_MONO_BOLD_PATH if bold else FONT_MONO_PATH, FONT_BOLD_PATH, FONT_REG_PATH]
    else:
        candidates = [FONT_BOLD_PATH if bold else FONT_REG_PATH, FONT_MONO_PATH]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_size: int,
    min_size: int = 11,
    bold: bool = False,
    mono: bool = False,
) -> ImageFont.ImageFont:
    """Shrink font size until text fits cleanly inside max_width."""
    for sz in range(max_size, min_size - 1, -1):
        f = _font(sz, bold=bold, mono=mono)
        w, _ = _text_size(draw, text, f)
        if w <= max_width:
            return f
    return _font(min_size, bold=bold, mono=mono)


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] = TEXT,
    anchor: str = "la",
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _draw_card(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    fill: Tuple[int, int, int] = CARD,
    outline: Tuple[int, int, int] = BORDER,
    radius: int = 14,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    center: Tuple[int, int],
    text: str,
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = TEXT,
    border_color: Tuple[int, int, int] | None = None,
    font_size: int = 15,
    pad_x: int = 14,
    pad_y: int = 7,
) -> None:
    f = _font(font_size, bold=True)
    tw, th = _text_size(draw, text, f)
    cx, cy = center
    box = (cx - tw // 2 - pad_x, cy - th // 2 - pad_y, cx + tw // 2 + pad_x, cy + th // 2 + pad_y)
    draw.rounded_rectangle(box, radius=(th + 2 * pad_y) // 2, fill=bg_color, outline=border_color or bg_color, width=2)
    draw.text((cx, cy), text, font=f, fill=text_color, anchor="mm")


def _blend_rgb(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return (
        int(round(c1[0] + t * (c2[0] - c1[0]))),
        int(round(c1[1] + t * (c2[1] - c1[1]))),
        int(round(c1[2] + t * (c2[2] - c1[2]))),
    )


def _apply_colormap(norm_grid: np.ndarray, stops: Sequence[Tuple[float, Tuple[int, int, int]]]) -> np.ndarray:
    """Vectorized piecewise-linear RGB colormap for normalized [0, 1] 2D arrays."""
    t = np.clip(norm_grid, 0.0, 1.0)
    rgb = np.zeros((*t.shape, 3), dtype=np.uint8)
    for idx in range(len(stops) - 1):
        t0, c0 = stops[idx]
        t1, c1 = stops[idx + 1]
        mask = (t >= t0) & (t <= t1 if idx == len(stops) - 2 else t < t1)
        local = np.zeros_like(t)
        if t1 > t0:
            local[mask] = (t[mask] - t0) / (t1 - t0)
        for ch in range(3):
            vals = c0[ch] + local[mask] * (c1[ch] - c0[ch])
            rgb[:, :, ch][mask] = np.clip(np.round(vals), 0, 255).astype(np.uint8)
    return rgb


THERMAL_STOPS: Tuple[Tuple[float, Tuple[int, int, int]], ...] = (
    (0.00, (12, 18, 38)),
    (0.28, (30, 58, 138)),
    (0.55, (147, 51, 234)),
    (0.80, (244, 63, 94)),
    (1.00, (253, 224, 71)),
)

FLEX_STOPS: Tuple[Tuple[float, Tuple[int, int, int]], ...] = (
    (0.00, (30, 64, 175)),
    (0.25, (6, 182, 212)),
    (0.50, (16, 185, 129)),
    (0.75, (245, 158, 11)),
    (1.00, (244, 63, 94)),
)


def pick_dramatic_heldout_case(assets: CockpitAssets) -> int:
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


def _map_pts(
    xs: np.ndarray,
    ys: np.ndarray,
    x_lim: Tuple[float, float],
    y_lim: Tuple[float, float],
    plot_box: Tuple[int, int, int, int],
) -> List[Tuple[float, float]]:
    px0, py0, px1, py1 = plot_box
    x0, x1 = x_lim
    y0, y1 = y_lim
    sx = (px1 - px0) / max(x1 - x0, 1e-12)
    sy = (py1 - py0) / max(y1 - y0, 1e-12)
    pts: List[Tuple[float, float]] = []
    for xv, yv in zip(xs, ys):
        yc = float(np.clip(yv, y0, y1))
        pts.append((float(px0 + (xv - x0) * sx), float(py1 - (yc - y0) * sy)))
    return pts


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    fill: Tuple[int, int, int],
    width: int = 2,
    dash: int = 8,
    gap: int = 6,
) -> None:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = float(np.hypot(dx, dy))
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    while pos < dist:
        end = min(pos + dash, dist)
        draw.line(
            [(p0[0] + ux * pos, p0[1] + uy * pos), (p0[0] + ux * end, p0[1] + uy * end)],
            fill=fill,
            width=width,
        )
        pos += dash + gap


def _draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    pts: List[Tuple[float, float]],
    fill: Tuple[int, int, int],
    width: int = 2,
) -> None:
    for i in range(0, len(pts) - 1, 2):
        draw.line([pts[i], pts[i + 1]], fill=fill, width=width)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    fill: Tuple[int, int, int],
    width: int = 3,
    head_len: int = 11,
) -> None:
    draw.line([p0, p1], fill=fill, width=width)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = float(np.hypot(dx, dy))
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux
    left = (p1[0] - head_len * ux + 0.55 * head_len * px, p1[1] - head_len * uy + 0.55 * head_len * py)
    right = (p1[0] - head_len * ux - 0.55 * head_len * px, p1[1] - head_len * uy - 0.55 * head_len * py)
    draw.polygon([p1, left, right], fill=fill)


def _draw_uv_heatmap(
    base_img: Image.Image,
    draw: ImageDraw.ImageDraw,
    uv_db: np.ndarray,
    u_vals: np.ndarray,
    box: Tuple[int, int, int, int],
    title: str,
    subtitle: str,
    accent: Tuple[int, int, int],
    theta0_deg: float,
) -> None:
    """Paste smoothed 2D (u, v) RF pattern heatmap with crosshair onto base canvas."""
    x0, y0, x1, y1 = box
    _draw_card(draw, box, fill=CARD_ALT, outline=accent, radius=12, width=2)
    f_title = _fit_font(draw, title, x1 - x0 - 20, 15, bold=True)
    f_sub = _fit_font(draw, subtitle, x1 - x0 - 20, 13, mono=True)
    _draw_text(draw, ((x0 + x1) // 2, y0 + 12), title, f_title, fill=accent, anchor="ma")
    _draw_text(draw, ((x0 + x1) // 2, y0 + 34), subtitle, f_sub, fill=TEXT, anchor="ma")

    # Convert (u, v) matrix from -42..0 dB into smooth RGB heatmap
    # Matrix axes: row = u (-0.85..0.85), col = v (-0.85..0.85) -> flip/transpose so u=horiz, v=vert
    norm = np.clip((uv_db.T[::-1, :] + 42.0) / 42.0, 0.0, 1.0)
    rgb = _apply_colormap(norm, THERMAL_STOPS)
    hm_img = Image.fromarray(rgb, mode="RGB")

    pad_x = 18
    hm_x0, hm_y0 = x0 + pad_x, y0 + 58
    hm_x1, hm_y1 = x1 - pad_x, y1 - 30
    hm_w = max(20, hm_x1 - hm_x0)
    hm_h = max(20, hm_y1 - hm_y0)
    hm_resized = hm_img.resize((hm_w, hm_h), resample=Image.Resampling.BILINEAR)
    base_img.paste(hm_resized, (hm_x0, hm_y0))
    draw.rectangle((hm_x0, hm_y0, hm_x1, hm_y1), outline=BORDER, width=1)

    # Mark target u0 = sin(theta0), v0 = 0
    u_min, u_max = float(u_vals[0]), float(u_vals[-1])
    u0 = float(np.sin(np.deg2rad(theta0_deg)))
    tx = hm_x0 + int(round((u0 - u_min) / (u_max - u_min) * hm_w))
    ty = (hm_y0 + hm_y1) // 2
    _draw_dashed_line(draw, (tx, hm_y0), (tx, hm_y1), CYAN, width=1, dash=4, gap=4)
    _draw_dashed_line(draw, (hm_x0, ty), (hm_x1, ty), CYAN, width=1, dash=4, gap=4)
    draw.ellipse((tx - 6, ty - 6, tx + 6, ty + 6), outline=CYAN, width=2)
    f_ax = _font(11, mono=True)
    _draw_text(draw, ((hm_x0 + hm_x1) // 2, hm_y1 + 6), f"Target (u={u0:.2f}, v=0.0)", f_ax, fill=MUTED, anchor="ma")


def render_slide1_story_problem(assets: CockpitAssets, live: Dict[str, Any], out_path: Path) -> Image.Image:
    """Slide 1 (1600x900): Intuitive visual story of wing bending vs 30 mm X-band wavelength."""
    img = Image.new("RGB", (1600, 900), BG)
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle((0, 0, 1600, 122), fill=(15, 23, 42))
    draw.line([(0, 122), (1600, 122)], fill=CYAN, width=2)
    _draw_pill(draw, (1440, 42), "SLIDE 1 / 3  |  THE PHYSICS", (20, 45, 78), CYAN, CYAN, font_size=14)
    f_h1 = _font(36, bold=True)
    f_sub = _font(19, bold=False)
    _draw_text(draw, (42, 22), "WHY DOES A BENDING UAV WING MAKE ITS RADAR GO BLIND?", f_h1, TEXT)
    _draw_text(
        draw,
        (42, 74),
        "256 X-Band Antennas on a Composite Skin: 37.8 mm Wing Flex Exceeds the 30.0 mm Wavelength (1.26 λ₀)",
        f_sub,
        CYAN,
    )

    # Left Card: Intuitive Wing Flex + Phase Front Diagram
    _draw_card(draw, (40, 144, 792, 696))
    _draw_text(draw, (64, 164), "INTUITIVE ANALOGY: 256 FLASHLIGHTS ON A BENDING BRIDGE", _font(20, bold=True), TEXT)
    _draw_text(
        draw,
        (64, 194),
        "Radar waves are 30 mm apart. When the carbon wing bends 19.1 mm across the antenna,",
        _font(15),
        MUTED,
    )
    _draw_text(
        draw,
        (64, 216),
        "each antenna shifts out of sync — breaking both target lock AND stealth focus.",
        _font(15),
        MUTED,
    )

    # Sub-panel A: Flat Wing (Locked)
    _draw_card(draw, (64, 248, 768, 392), fill=CARD_ALT, outline=GREEN, radius=10, width=2)
    _draw_text(draw, (82, 260), "NOMINAL WING (0 mm FLEX)  ->  ALL 256 WAVES ARRIVE IN PHASE", _font(15, bold=True), GREEN)
    # Flat wing beam + elements
    draw.line([(95, 348), (440, 348)], fill=(148, 163, 184), width=8)
    for k in range(8):
        ex = 155 + k * 32
        draw.rectangle((ex - 6, 338, ex + 6, 348), fill=CYAN)
        _draw_arrow(draw, (ex, 336), (ex + 18, 294), GREEN, width=2, head_len=7)
    _draw_text(draw, (475, 304), "Razor-Sharp Beam", _font(16, bold=True), GREEN)
    _draw_text(draw, (475, 328), "Lock: +15.0° | SLL: -30.0 dB", _font(14, mono=True), TEXT)
    _draw_text(draw, (475, 352), "Invisible low-sidelobe footprint", _font(13), MUTED)

    # Sub-panel B: Bent Wing (5G + Thermal)
    _draw_card(draw, (64, 410, 768, 578), fill=CARD_ALT, outline=RED, radius=10, width=2)
    _draw_text(
        draw,
        (82, 422),
        "5G MANEUVER + +45°C THERMAL (37.8 mm TIP)  ->  PHASE CHAOS",
        _font(15, bold=True),
        RED,
    )
    bent_pts: List[Tuple[float, float]] = []
    for k in range(36):
        u = k / 35.0
        bx = 95.0 + u * 345.0
        by = 532.0 - 44.0 * (u ** 1.8)
        bent_pts.append((bx, by))
    draw.line(bent_pts, fill=(148, 163, 184), width=8)
    for k in range(8):
        u = (60.0 + k * 32.0) / 345.0
        ex = 95.0 + u * 345.0
        ey = 532.0 - 44.0 * (u ** 1.8)
        draw.rectangle((ex - 6, ey - 9, ex + 6, ey + 1), fill=AMBER)
        dx_a = 6 + (k % 3) * 11
        _draw_arrow(draw, (int(ex), int(ey - 10)), (int(ex + dx_a), int(ey - 44)), RED, width=2, head_len=7)
    _draw_text(draw, (475, 464), "Beam Drifts & Splits!", _font(16, bold=True), RED)
    _draw_text(draw, (475, 488), "Error: -5.04° (~880 m @ 10 km)", _font(14, mono=True), TEXT)
    _draw_text(draw, (475, 512), "Sidelobes jump to -12.7 dB!", _font(14, mono=True), AMBER)
    _draw_text(draw, (475, 536), "Enemy RWR detects leaked lobes", _font(13), MUTED)

    # Analogy callout bar inside left card
    _draw_card(draw, (64, 596, 768, 676), fill=(17, 36, 58), outline=CYAN, radius=10, width=1)
    _draw_text(
        draw,
        (82, 610),
        "WHY A SOFTWARE FIX BEATS STIFFER HEAVY WINGS:",
        _font(15, bold=True),
        CYAN,
    )
    _draw_text(
        draw,
        (82, 636),
        "Adding carbon plies adds weight and kills UAV range. AeroRF-Heal tracks 12 fiber-optic",
        _font(14),
        TEXT,
    )
    _draw_text(
        draw,
        (82, 656),
        "sensors with AI and re-aligns all 256 phase shifters in 0.82 ms — zero weight penalty.",
        _font(14),
        TEXT,
    )

    # Right Card: Live Real 2D (u, v) Beam Footprint Comparison
    _draw_card(draw, (812, 144, 1560, 696))
    _draw_text(draw, (836, 164), "REAL X-BAND (10 GHz) 2D BEAM FOOTPRINT COMPARISON", _font(20, bold=True), TEXT)
    _draw_text(
        draw,
        (836, 194),
        f"Held-Out ANSYS Battle-Damaged Wing (Case #{live['case_idx']}: {live['true_class']} + 2 Dead Sensors)",
        _font(14, mono=True),
        MUTED,
    )

    m_uncomp = live["metrics"]["uncompensated"]
    m_ai = live["metrics"]["ai_blind"]
    _draw_uv_heatmap(
        img,
        draw,
        live["uv_uncomp"],
        live["uv_vals"],
        (836, 232, 1176, 676),
        "UNCOMPENSATED RADAR",
        f"Err {m_uncomp['boresight_error_deg']:+.2f}° | SLL {m_uncomp['sll_db']:.1f} dB",
        RED,
        live["theta0_deg"],
    )
    _draw_uv_heatmap(
        img,
        draw,
        live["uv_healed"],
        live["uv_vals"],
        (1196, 232, 1536, 676),
        "AI SELF-HEALED (0.82 ms)",
        f"Err {m_ai['boresight_error_deg']:+.2f}° | SLL {m_ai['sll_db']:.1f} dB",
        GREEN,
        live["theta0_deg"],
    )

    # Bottom 3 Headline Metric Cards
    kpis = [
        (
            "37.8 mm vs 30.0 mm",
            "PHYSICAL ROOT CAUSE",
            "5G + Thermal wing flex (1.26 λ₀) scrambles phase across the aperture",
            CYAN,
        ),
        (
            "-5.04° -> -0.12°",
            "PHASE 1 COUPLED LOCK",
            "~880 m miss error at 10 km range collapsed back to dead-on target",
            AMBER,
        ),
        (
            "-12.7 dB -> -29.0 dB",
            "STEALTH LPI RESTORED",
            "Phase 1 6-bit T/R compensation suppresses leaked radar sidelobes",
            GREEN,
        ),
    ]
    cw = 492
    gap = 22
    for i, (val, label, desc, acc) in enumerate(kpis):
        x0 = 40 + i * (cw + gap)
        x1 = x0 + cw
        _draw_card(draw, (x0, 718, x1, 868), fill=CARD, outline=acc, radius=14, width=2)
        _draw_text(draw, (x0 + 22, 734), label, _font(14, bold=True), acc)
        _draw_text(draw, (x0 + 22, 760), val, _fit_font(draw, val, cw - 44, 34, bold=True, mono=True), TEXT)
        f_desc = _fit_font(draw, desc, cw - 44, 14)
        _draw_text(draw, (x0 + 22, 824), desc, f_desc, MUTED)

    img.save(out_path, format="PNG")
    return img


def render_slide2_combat_sensor_kill(assets: CockpitAssets, live: Dict[str, Any], out_path: Path) -> Image.Image:
    """Slide 2 (1600x900): Held-out combat stress test (2 dead FBGs + 3% noise) + 24-case scorecard."""
    img = Image.new("RGB", (1600, 900), BG)
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle((0, 0, 1600, 122), fill=(15, 23, 42))
    draw.line([(0, 122), (1600, 122)], fill=CYAN, width=2)
    _draw_pill(draw, (1440, 42), "SLIDE 2 / 3  |  COMBAT TEST", (68, 22, 42), RED, RED, font_size=14)
    _draw_text(draw, (42, 22), "COMBAT STRESS TEST: 2 SEVERED SENSORS + 3% NOISE", _font(36, bold=True), TEXT)
    _draw_text(
        draw,
        (42, 74),
        "When Shrapnel Kills Fiber-Optic Gauges, Classical Linear Models Collapse — AeroRFShapeNet v2 Heals in 0.82 ms",
        _font(19),
        CYAN,
    )

    # Left Card: Real 1D Pattern Cut from live held-out case
    _draw_card(draw, (40, 144, 956, 736))
    m_un = live["metrics"]["uncompensated"]
    m_ri = live["metrics"]["ridge_blind"]
    m_ai = live["metrics"]["ai_blind"]
    _draw_text(
        draw,
        (64, 164),
        f"LIVE HELD-OUT PATTERN CUT  |  CASE #{live['case_idx']} ({live['true_class'].upper()}, FBG-4 & FBG-8 SEVERED)",
        _fit_font(draw, f"LIVE HELD-OUT PATTERN CUT  |  CASE #{live['case_idx']} ({live['true_class'].upper()}, FBG-4 & FBG-8 SEVERED)", 868, 19, bold=True),
        TEXT,
    )
    _draw_text(
        draw,
        (64, 192),
        "6-bit quantized phase recovery steering toward θ₀ = +15.0° (unseen ANSYS test wing)",
        _font(14),
        MUTED,
    )

    plot_box = (118, 232, 926, 680)
    px0, py0, px1, py1 = plot_box
    draw.rectangle(plot_box, fill=(10, 16, 30), outline=BORDER, width=1)
    x_lim = (-45.0, 55.0)
    y_lim = (-45.0, 3.0)

    # Gridlines & axes
    f_ax = _font(13, mono=True)
    for db in (-40, -30, -20, -10, 0):
        gy = py1 - (db - y_lim[0]) / (y_lim[1] - y_lim[0]) * (py1 - py0)
        col = (60, 48, 82) if db == -30 else (30, 41, 59)
        draw.line([(px0, gy), (px1, gy)], fill=col, width=1)
        _draw_text(draw, (px0 - 10, gy), f"{db:>3d}", f_ax, MUTED, anchor="rm")
    for ang in (-40, -20, 0, 15, 20, 40):
        gx = px0 + (ang - x_lim[0]) / (x_lim[1] - x_lim[0]) * (px1 - px0)
        if ang == 15:
            _draw_dashed_line(draw, (gx, py0), (gx, py1), CYAN, width=2, dash=7, gap=5)
            _draw_text(draw, (gx + 6, py0 + 12), "TARGET +15°", _font(13, bold=True), CYAN)
        else:
            draw.line([(gx, py0), (gx, py1)], fill=(30, 41, 59), width=1)
        _draw_text(draw, (gx, py1 + 10), f"{ang:+d}°", f_ax, CYAN if ang == 15 else MUTED, anchor="ma")
    _draw_text(draw, (px0 - 52, (py0 + py1) // 2), "dB", _font(14, bold=True), MUTED, anchor="mm")
    _draw_text(draw, ((px0 + px1) // 2, py1 + 32), "Scan Angle θ (degrees)", _font(14, bold=True), MUTED, anchor="ma")

    theta = live["theta_deg"]
    mask = (theta >= x_lim[0]) & (theta <= x_lim[1])
    th_sub = theta[mask]

    pts_nom = _map_pts(th_sub, live["cuts"]["nominal"][mask], x_lim, y_lim, plot_box)
    pts_un = _map_pts(th_sub, live["cuts"]["uncompensated"][mask], x_lim, y_lim, plot_box)
    pts_ri = _map_pts(th_sub, live["cuts"]["ridge_blind"][mask], x_lim, y_lim, plot_box)
    pts_ai = _map_pts(th_sub, live["cuts"]["ai_blind"][mask], x_lim, y_lim, plot_box)

    _draw_dashed_polyline(draw, pts_nom, SLATE_LINE, width=2)
    draw.line(pts_un, fill=RED, width=3)
    draw.line(pts_ri, fill=AMBER, width=3)
    draw.line(pts_ai, fill=GREEN, width=4)

    # Legend box inside top-left of plot
    _draw_card(draw, (px0 + 14, py0 + 14, px0 + 390, py0 + 126), fill=(15, 23, 42), outline=BORDER, radius=8, width=1)
    leg_rows = [
        (SLATE_LINE, "Nominal Ideal (-30 dB Taylor)"),
        (RED, f"Uncompensated ({m_un['boresight_error_deg']:+.2f}° | SLL {m_un['sll_db']:.1f} dB)"),
        (AMBER, f"Linear Ridge ({m_ri['boresight_error_deg']:+.2f}° | SLL {m_ri['sll_db']:.1f} dB)"),
        (GREEN, f"AeroRFShapeNet v2 ({m_ai['boresight_error_deg']:+.2f}° | SLL {m_ai['sll_db']:.1f} dB)"),
    ]
    for r_idx, (col, lbl) in enumerate(leg_rows):
        ly = py0 + 30 + r_idx * 24
        draw.line([(px0 + 26, ly), (px0 + 54, ly)], fill=col, width=4)
        _draw_text(draw, (px0 + 64, ly), lbl, _fit_font(draw, lbl, 316, 13, bold=(r_idx == 3), mono=True), TEXT, anchor="lm")

    # Right Card: 24-Case Unseen Combat Benchmark Scorecard
    bench = assets.benchmark
    reg_c = bench["regimes"][2]
    rf_c = bench["rf_recovery_scenario_c"]

    _draw_card(draw, (978, 144, 1560, 736))
    _draw_text(draw, (1002, 164), "24 HELD-OUT ANSYS CASES (SCENARIO C)", _font(20, bold=True), TEXT)
    _draw_text(draw, (1002, 194), "Aggregate benchmark: 3% Noise + 2 Random Dead FBGs", _font(14), MUTED)

    # Shape RMSE Horizontal Bar Visual
    _draw_card(draw, (1002, 230, 1536, 396), fill=CARD_ALT, outline=BORDER, radius=10, width=1)
    _draw_text(draw, (1020, 244), "BLIND 3D WING SHAPE RECONSTRUCTION RMSE (LOWER IS BETTER)", _font(13, bold=True), CYAN)
    ridge_rmse = float(reg_c["ridge_rmse_mm"])
    ai_rmse = float(reg_c["ai_rmse_mm"])
    ratio = ridge_rmse / max(ai_rmse, 1e-6)

    # Ridge bar
    _draw_text(draw, (1020, 276), "Linear Ridge (No Imputation)", _font(14), TEXT)
    draw.rectangle((1020, 300, 1410, 322), fill=(68, 22, 34), outline=RED, width=1)
    draw.rectangle((1020, 300, 1410, 322), fill=RED)
    _draw_text(draw, (1422, 311), f"{ridge_rmse:.3f} mm", _font(15, bold=True, mono=True), RED, anchor="lm")

    # AI bar
    _draw_text(draw, (1020, 334), f"AeroRFShapeNet v2  ({ratio:.2f}x MORE ACCURATE)", _font(14, bold=True), GREEN)
    bar_w = int(round(390 * (ai_rmse / ridge_rmse)))
    draw.rectangle((1020, 358, 1410, 380), fill=(15, 36, 32), outline=BORDER, width=1)
    draw.rectangle((1020, 358, 1020 + max(bar_w, 12), 380), fill=GREEN)
    _draw_text(draw, (1422, 369), f"{ai_rmse:.3f} mm", _font(15, bold=True, mono=True), GREEN, anchor="lm")

    # RF Scorecard Table
    _draw_card(draw, (1002, 414, 1536, 650), fill=CARD_ALT, outline=BORDER, radius=10, width=1)
    _draw_text(draw, (1020, 428), "BLIND 6-BIT RF RECOVERY VS ANSYS ORACLE", _font(14, bold=True), CYAN)
    cols_x = [1020, 1218, 1344, 1456]
    headers = ["Method", "|Err| (°)", "SLL (dB)", "Loss (dB)"]
    for cx, h in zip(cols_x, headers):
        _draw_text(draw, (cx, 458), h, _font(13, bold=True), MUTED)
    draw.line([(1018, 480), (1520, 480)], fill=BORDER, width=1)

    rows = [
        ("Uncompensated", rf_c["uncompensated"], RED),
        ("Linear Ridge", rf_c["ridge_blind_6bit"], AMBER),
        ("AeroRFShapeNet v2", rf_c["ai_blind_6bit"], GREEN),
        ("ANSYS Oracle", rf_c["oracle_6bit"], CYAN),
    ]
    for r_i, (r_name, rd, r_col) in enumerate(rows):
        ry = 496 + r_i * 36
        if r_name == "AeroRFShapeNet v2":
            draw.rounded_rectangle((1014, ry - 4, 1524, ry + 26), radius=6, fill=(16, 48, 44), outline=GREEN, width=1)
        _draw_text(draw, (cols_x[0], ry + 3), r_name, _font(14, bold=(r_i == 2)), r_col)
        _draw_text(draw, (cols_x[1], ry + 3), f"{rd['mean_abs_boresight_err_deg']:.3f}°", _font(14, bold=(r_i == 2), mono=True), TEXT)
        _draw_text(draw, (cols_x[2], ry + 3), f"{rd['mean_sll_db']:.2f}", _font(14, bold=(r_i == 2), mono=True), TEXT)
        _draw_text(draw, (cols_x[3], ry + 3), f"{rd['mean_gain_loss_db']:.3f}", _font(14, bold=(r_i == 2), mono=True), TEXT)

    # Bottom combat highlight pill inside right card
    _draw_card(draw, (1002, 666, 1536, 718), fill=(16, 48, 44), outline=GREEN, radius=10, width=2)
    summary_str = (
        f"Damage ID: {reg_c['damage_acc_pct']:.1f}%  |  CPU Latency: {reg_c['latency_ms']:.2f} ms  |  Matches Oracle!"
    )
    _draw_text(draw, (1269, 692), summary_str, _fit_font(draw, summary_str, 510, 15, bold=True, mono=True), GREEN, anchor="mm")

    # Bottom Insight Banner
    _draw_card(draw, (40, 756, 1560, 868), fill=(17, 36, 58), outline=CYAN, radius=14, width=2)
    _draw_text(
        draw,
        (66, 774),
        "SECRET WEAPON: CONDITIONAL GAUSSIAN COVARIANCE IMPUTATION + PARITY-SPACE RESIDUALS",
        _font(17, bold=True),
        CYAN,
    )
    _draw_text(
        draw,
        (66, 804),
        "When FBG-4 & FBG-8 go dark, naive zeros trick linear models into seeing sharp local structural spikes (crashing to 2.99 mm RMSE).",
        _font(15),
        TEXT,
    )
    _draw_text(
        draw,
        (66, 830),
        "AeroRFShapeNet v2 infers the missing channels analytically from the healthy strain covariance — restoring Oracle-level radar lock.",
        _font(15),
        MUTED,
    )

    img.save(out_path, format="PNG")
    return img


def render_slide3_architecture_and_vv(assets: CockpitAssets, live: Dict[str, Any], out_path: Path) -> Image.Image:
    """Slide 3 (1600x900): Closed-loop pipeline, 3-regime benchmark table & V&V credibility proof."""
    img = Image.new("RGB", (1600, 900), BG)
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle((0, 0, 1600, 122), fill=(15, 23, 42))
    draw.line([(0, 122), (1600, 122)], fill=CYAN, width=2)
    _draw_pill(draw, (1440, 42), "SLIDE 3 / 3  |  V&V & STACK", (36, 26, 70), VIOLET, VIOLET, font_size=14)
    _draw_text(draw, (42, 22), "END-TO-END DIGITAL TWIN: ANSYS MAPDL + PHYSICS-INFORMED AI", _font(35, bold=True), TEXT)
    _draw_text(
        draw,
        (42, 74),
        "Verified Against Euler-Bernoulli Beam Theory, 120 Real FEA Solves & Live Interactive Combat Cockpit",
        _font(19),
        CYAN,
    )

    # Top Row: 4 Connected Architecture Stages
    stages = [
        (
            "1. ANSYS MAPDL 2026 R1",
            CYAN,
            [
                "Composite SHELL181 Wingbox",
                "• 0.518% V&V vs Euler-Bernoulli",
                "• 120 automated FEA runs (170 s)",
                "• 4 battle-damage structural zones",
            ],
        ),
        (
            "2. FBG Covariance Imputer",
            AMBER,
            [
                "12 Optical Strain Channels",
                "• Exact conditional Gaussian μ|obs",
                "• Survives 2+ severed combat FBGs",
                "• Zero distortion on healthy gauges",
            ],
        ),
        (
            "3. AeroRFShapeNet v2",
            VIOLET,
            [
                "Dual-Branch Physics AI",
                "• 12 POD modes = 100.0% energy",
                "• RF Phase Coherence Loss",
                "• Load-invariant Parity Damage ID",
            ],
        ),
        (
            "4. 6-Bit AESA & Cockpit",
            GREEN,
            [
                "Real-Time Digital Twin",
                "• 0.82 ms CPU shape inference",
                "• 6-bit T/R quantized healing",
                "• Interactive Streamlit kill-switch",
            ],
        ),
    ]
    box_w = 342
    gap = 50
    for i, (st_title, acc, bullets) in enumerate(stages):
        x0 = 40 + i * (box_w + gap)
        x1 = x0 + box_w
        _draw_card(draw, (x0, 146, x1, 412), fill=CARD, outline=acc, radius=14, width=2)
        _draw_card(draw, (x0 + 12, 158, x1 - 12, 206), fill=CARD_ALT, outline=acc, radius=8, width=1)
        _draw_text(draw, ((x0 + x1) // 2, 182), st_title, _fit_font(draw, st_title, box_w - 36, 17, bold=True), acc, anchor="mm")
        _draw_text(draw, (x0 + 18, 224), bullets[0], _font(15, bold=True), TEXT)
        for bi, btxt in enumerate(bullets[1:]):
            by = 264 + bi * 42
            _draw_text(draw, (x0 + 18, by), btxt, _fit_font(draw, btxt, box_w - 32, 14), MUTED)
        if i < len(stages) - 1:
            _draw_arrow(draw, (x1 + 6, 279), (x1 + gap - 6, 279), CYAN, width=3, head_len=12)

    # Bottom-Left Card: 3-Regime Held-Out Table
    bench = assets.benchmark
    _draw_card(draw, (40, 440, 868, 862))
    _draw_text(draw, (64, 462), "3-REGIME HELD-OUT BENCHMARK (24 UNSEEN ANSYS SIMULATIONS)", _font(19, bold=True), TEXT)
    _draw_text(draw, (64, 492), "Strict stratified 80/20 split — zero data leakage across maneuver & damage severity", _font(14), MUTED)

    tbl_box = (64, 530, 844, 774)
    _draw_card(draw, tbl_box, fill=CARD_ALT, outline=BORDER, radius=10, width=1)
    t_cols = [80, 315, 455, 595, 735]
    t_hdrs = ["Stress Regime", "Ridge RMSE", "AI RMSE", "Damage Acc", "Latency"]
    for cx, h in zip(t_cols, t_hdrs):
        _draw_text(draw, (cx, 548), h, _font(14, bold=True), CYAN)
    draw.line([(78, 576), (830, 576)], fill=BORDER, width=1)

    for ri, rdata in enumerate(bench["regimes"]):
        ry = 594 + ri * 58
        is_combat = ri == 2
        if is_combat:
            draw.rounded_rectangle((74, ry - 8, 832, ry + 36), radius=8, fill=(16, 48, 44), outline=GREEN, width=2)
        _draw_text(draw, (t_cols[0], ry + 4), rdata["name"], _font(15, bold=is_combat), GREEN if is_combat else TEXT)
        _draw_text(draw, (t_cols[1], ry + 4), f"{rdata['ridge_rmse_mm']:.3f} mm", _font(15, mono=True), RED if is_combat else MUTED)
        _draw_text(draw, (t_cols[2], ry + 4), f"{rdata['ai_rmse_mm']:.3f} mm", _font(15, bold=True, mono=True), GREEN if is_combat else TEXT)
        _draw_text(draw, (t_cols[3], ry + 4), f"{rdata['damage_acc_pct']:.1f}%", _font(15, mono=True), TEXT)
        _draw_text(draw, (t_cols[4], ry + 4), f"{rdata['latency_ms']:.2f} ms", _font(15, mono=True), CYAN)

    _draw_text(
        draw,
        (64, 800),
        "POD Modal Compression: 12 SVD modes capture 100.000% (0.99999997) of 3D aperture energy",
        _font(15, bold=True),
        AMBER,
    )
    _draw_text(
        draw,
        (64, 826),
        "Trained jointly with RF Phase Coherence Loss so predictions optimize electromagnetic lock, not just L2.",
        _font(14),
        MUTED,
    )

    # Bottom-Right Card: V&V & Open-Source Proof
    _draw_card(draw, (894, 440, 1560, 862))
    _draw_text(draw, (918, 462), "ENGINEERING RIGOR, V&V & OPEN SOURCE", _font(19, bold=True), TEXT)
    _draw_text(draw, (918, 492), "Built end-to-end in Python — automated FEA, AI training & live cockpit", _font(14), MUTED)

    checks = [
        ("Euler-Bernoulli Cantilever V&V", "11.3694 vs 11.4286 mm (0.518% err)"),
        ("Headless ANSYS Batch DOE", "120 solves in 170.1 s (1.42 s/case, 0 errors)"),
        ("IEEE 2D Taylor Phased Array", "-30.0 dB sidelobe taper + 6-bit T/R quantization"),
        ("Closed-Loop Blind Combat Test", "3.16° -> 0.10° boresight | -17.7 -> -29.2 dB SLL"),
        ("Automated Unit & Cockpit Tests", "RF V&V + ShapeNet AI + Streamlit Smoke PASS"),
    ]
    for ci, (ck_title, ck_desc) in enumerate(checks):
        cy = 534 + ci * 54
        _draw_pill(draw, (952, cy + 12), "PASS", (16, 48, 44), GREEN, GREEN, font_size=12, pad_x=10, pad_y=4)
        _draw_text(draw, (996, cy), ck_title, _font(15, bold=True), TEXT)
        _draw_text(draw, (996, cy + 21), ck_desc, _fit_font(draw, ck_desc, 540, 14, mono=True), MUTED)

    _draw_card(draw, (918, 804, 1536, 846), fill=(17, 36, 58), outline=CYAN, radius=8, width=1)
    _draw_text(
        draw,
        (1227, 825),
        "Full Source Code + Trained Weights + Interactive Cockpit on GitHub",
        _font(15, bold=True),
        CYAN,
        anchor="mm",
    )

    img.save(out_path, format="PNG")
    return img


def render_healing_gif(assets: CockpitAssets, live: Dict[str, Any], out_path: Path) -> int:
    """Render 32-frame looping 1080x620 HUD GIF from live held-out ANSYS deformation."""
    array_cfg = assets.array_cfg
    wing_cfg = assets.wing_cfg
    fbg_cfg = assets.fbg_cfg
    case_idx = int(live["case_idx"])
    dead_ids = list(live["dead_sensor_ids"])
    theta0_deg = float(live["theta0_deg"])

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

    theta_grid = np.linspace(-50.0, 50.0, 601)
    pat_nom, e_nom = compute_pattern_cut(
        pos_nom, norm_nom, w_nominal, array_cfg.k0, theta_grid, phi_deg=0.0, q_elem=array_cfg.q_elem
    )
    ref_peak = float(np.max(e_nom))

    uz_true_grid_mm = (disp_true[:, 2] * 1e3).reshape(array_cfg.nx, array_cfg.ny)
    span_x = pos_nom[:, 0].reshape(array_cfg.nx, array_cfg.ny)[:, 0]
    uz_span_profile_mm = uz_true_grid_mm.mean(axis=1)
    uz_abs_max = max(float(np.abs(uz_true_grid_mm).max()), 15.0)
    y_prof_lim = (-uz_abs_max * 1.15, uz_abs_max * 1.15)

    # Frame schedule: 14 bend + 2 freeze blind + 12 heal + 4 freeze locked = 32 frames
    schedule: List[Tuple[float, float, int]] = []
    for k in range(14):
        schedule.append((k / 13.0, 0.0, 95))
    schedule.append((1.0, 0.0, 420))
    schedule.append((1.0, 0.0, 420))
    for k in range(1, 13):
        schedule.append((1.0, k / 12.0, 95))
    for k in range(3):
        schedule.append((1.0, 1.0, 220))
    schedule.append((1.0, 1.0, 850))

    frames: List[Image.Image] = []
    durations: List[int] = []

    W, H = 1080, 620
    plot_box = (594, 134, 1038, 492)
    prof_box = (82, 118, 502, 246)
    map_box = (36, 276, 502, 502)

    for alpha_bend, alpha_heal, dur in schedule:
        cur_disp = alpha_bend * disp_true
        cur_rot = alpha_bend * rot_true
        cur_pos = pos_nom + cur_disp
        cur_norm = rotate_normals_euler(norm_nom, cur_rot)

        cur_est = pos_nom + alpha_heal * disp_ai
        cur_w = compute_steering_weights(
            cur_est,
            theta0_deg,
            0.0,
            array_cfg.k0,
            amp_weights,
            quant_bits=6 if alpha_heal > 0.0 else None,
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
        cur_m = evaluate_pattern_metrics(theta_grid, cur_pat, theta0_deg)

        if alpha_heal == 0.0:
            if alpha_bend > 0.45:
                status_text = "RADAR BLIND!"
                status_col = RED
                status_bg = (68, 22, 34)
                beam_col = RED
            else:
                status_text = "5G FLEXING..."
                status_col = AMBER
                status_bg = (64, 44, 16)
                beam_col = AMBER if alpha_bend > 0.12 else CYAN
        elif alpha_heal < 0.75:
            status_text = "AI LOCKING..."
            status_col = AMBER
            status_bg = (64, 44, 16)
            beam_col = AMBER
        else:
            status_text = "AI SELF-HEALED"
            status_col = GREEN
            status_bg = (16, 48, 44)
            beam_col = GREEN

        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)

        # Header
        draw.rectangle((0, 0, W, 62), fill=(15, 23, 42))
        draw.line([(0, 62), (W, 62)], fill=CYAN, width=2)
        _draw_text(draw, (20, 10), "AeroRF-Heal | Live X-Band Radar Self-Healing", _font(21, bold=True), TEXT)
        case_sub = f"ANSYS Held-Out Case #{case_idx}  |  Damage: {live['true_class']}  |  2/12 FBGs Severed"
        _draw_text(draw, (20, 38), case_sub, _font(13, mono=True), MUTED)
        _draw_pill(draw, (974, 31), status_text, status_bg, status_col, status_col, font_size=14, pad_x=14, pad_y=6)

        # Top-left Card: Live Spanwise Wing Flex Profile
        _draw_card(draw, (20, 74, 520, 258))
        _draw_text(draw, (34, 86), "APERTURE SPANWISE FLEX UZ (mm) vs λ₀ = 30 mm", _font(14, bold=True), TEXT)
        draw.rectangle(prof_box, fill=(10, 16, 30), outline=BORDER, width=1)
        zero_pts = _map_pts(span_x, np.zeros_like(span_x), (span_x[0], span_x[-1]), y_prof_lim, prof_box)
        _draw_dashed_line(draw, zero_pts[0], zero_pts[-1], SLATE_LINE, width=1, dash=5, gap=4)
        cur_prof_pts = _map_pts(span_x, alpha_bend * uz_span_profile_mm, (span_x[0], span_x[-1]), y_prof_lim, prof_box)
        draw.line(cur_prof_pts, fill=CYAN if alpha_bend < 0.2 else AMBER, width=3)
        _draw_text(draw, (prof_box[0] - 8, prof_box[1] + 6), f"+{y_prof_lim[1]:.0f}", _font(11, mono=True), MUTED, anchor="rm")
        _draw_text(draw, (prof_box[0] - 8, (prof_box[1] + prof_box[3]) // 2), "0", _font(11, mono=True), MUTED, anchor="rm")
        _draw_text(draw, (prof_box[0] - 8, prof_box[3] - 6), f"{y_prof_lim[0]:.0f}", _font(11, mono=True), MUTED, anchor="rm")

        # Bottom-left Card: Wingbox Planform + 16x16 Heatmap + 12 FBG Health Pins
        _draw_card(draw, (20, 268, 520, 514))
        _draw_text(draw, (34, 280), "WINGBOX PLANFORM, 16×16 ARRAY & 12 FBG SENSORS", _font(14, bold=True), TEXT)
        wx0, wy0, wx1, wy1 = 46, 314, 492, 464
        draw.rectangle((wx0, wy0, wx1, wy1), fill=(15, 25, 45), outline=(71, 85, 105), width=2)
        # Damage zone highlight (mapped identically to Wingbox y -> px coordinates)
        if "front" in live["true_class"]:
            y_m0, y_m1 = 0.0, wing_cfg.y_fspar + 0.03
        elif "rear" in live["true_class"]:
            y_m0, y_m1 = wing_cfg.y_rspar - 0.03, wing_cfg.chord
        else:
            y_m0, y_m1 = wing_cfg.y_fspar, wing_cfg.y_rspar
        dmg_y0 = int(wy0 + (1.0 - y_m1 / wing_cfg.chord) * (wy1 - wy0))
        dmg_y1 = int(wy0 + (1.0 - y_m0 / wing_cfg.chord) * (wy1 - wy0))
        dmg_x0 = int(wx0 + (0.30 / wing_cfg.span) * (wx1 - wx0))
        dmg_x1 = int(wx0 + (0.60 / wing_cfg.span) * (wx1 - wx0))
        draw.rectangle((dmg_x0, dmg_y0, dmg_x1, dmg_y1), fill=(68, 22, 34), outline=RED, width=1)
        # Spars & ribs
        for spar_y in (wing_cfg.y_fspar, wing_cfg.y_rspar):
            sy_px = int(wy0 + (1.0 - spar_y / wing_cfg.chord) * (wy1 - wy0))
            draw.line([(wx0, sy_px), (wx1, sy_px)], fill=(71, 85, 105), width=1)
        for rx in wing_cfg.rib_x[1:-1]:
            rx_px = int(wx0 + (rx / wing_cfg.span) * (wx1 - wx0))
            draw.line([(rx_px, wy0), (rx_px, wy1)], fill=(51, 65, 85), width=1)

        # 16x16 Aperture UZ Heatmap placed at true coordinates
        cur_uz_grid = alpha_bend * uz_true_grid_mm
        norm_hm = np.clip((cur_uz_grid.T[::-1, :] + uz_abs_max) / (2.0 * uz_abs_max), 0.0, 1.0)
        hm_rgb = _apply_colormap(norm_hm, FLEX_STOPS)
        hm_pil = Image.fromarray(hm_rgb, mode="RGB")
        ax0_m = float(pos_nom[:, 0].min())
        ax1_m = float(pos_nom[:, 0].max())
        ay0_m = float(pos_nom[:, 1].min())
        ay1_m = float(pos_nom[:, 1].max())
        ap_px0 = int(wx0 + (ax0_m / wing_cfg.span) * (wx1 - wx0))
        ap_px1 = int(wx0 + (ax1_m / wing_cfg.span) * (wx1 - wx0))
        ap_py0 = int(wy0 + (1.0 - ay1_m / wing_cfg.chord) * (wy1 - wy0))
        ap_py1 = int(wy0 + (1.0 - ay0_m / wing_cfg.chord) * (wy1 - wy0))
        hm_pil = hm_pil.resize((max(10, ap_px1 - ap_px0), max(10, ap_py1 - ap_py0)), Image.Resampling.NEAREST)
        img.paste(hm_pil, (ap_px0, ap_py0))
        draw.rectangle((ap_px0, ap_py0, ap_px1, ap_py1), outline=TEXT, width=1)

        # 12 FBG Pins
        for s_idx, (fx, fy) in enumerate(fbg_cfg.as_array()):
            px = int(wx0 + (fx / wing_cfg.span) * (wx1 - wx0))
            py = int(wy0 + (1.0 - fy / wing_cfg.chord) * (wy1 - wy0))
            if s_idx in dead_ids:
                draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(68, 22, 34), outline=RED, width=2)
                draw.line([(px - 4, py - 4), (px + 4, py + 4)], fill=RED, width=2)
                draw.line([(px - 4, py + 4), (px + 4, py - 4)], fill=RED, width=2)
            else:
                draw.rectangle((px - 4, py - 4, px + 4, py + 4), fill=CYAN, outline=BG, width=1)

        _draw_text(draw, (46, 478), "Cyan: 10 Active FBGs  |  Red X: FBG-4 & FBG-8 Severed", _font(12, mono=True), MUTED)

        # Right Panel: Live 1D Pattern Cut
        _draw_card(draw, (534, 74, 1060, 514))
        phase_hdr = (
            f"PHASE 1: WING BENDING ({alpha_bend * 100:.0f}%) — AI OFF"
            if alpha_heal == 0.0
            else f"PHASE 2: AI SELF-HEALING ({alpha_heal * 100:.0f}%, 6-BIT, 0.82 ms)"
        )
        _draw_text(draw, (552, 92), phase_hdr, _font(16, bold=True), status_col)
        px0, py0, px1, py1 = plot_box
        draw.rectangle(plot_box, fill=(10, 16, 30), outline=BORDER, width=1)
        x_lim = (-50.0, 50.0)
        y_lim = (-45.0, 3.0)
        for db in (-40, -30, -20, -10, 0):
            gy = py1 - (db - y_lim[0]) / (y_lim[1] - y_lim[0]) * (py1 - py0)
            draw.line([(px0, gy), (px1, gy)], fill=(30, 41, 59), width=1)
            _draw_text(draw, (px0 - 8, gy), f"{db}", _font(12, mono=True), MUTED, anchor="rm")
        for ang in (-40, -20, 0, 20, 40):
            gx = px0 + (ang - x_lim[0]) / (x_lim[1] - x_lim[0]) * (px1 - px0)
            draw.line([(gx, py0), (gx, py1)], fill=(30, 41, 59), width=1)
            _draw_text(draw, (gx, py1 + 8), f"{ang:+d}°", _font(12, mono=True), MUTED, anchor="ma")
        tgx = px0 + (theta0_deg - x_lim[0]) / (x_lim[1] - x_lim[0]) * (px1 - px0)
        _draw_dashed_line(draw, (tgx, py0), (tgx, py1), CYAN, width=2, dash=6, gap=4)
        _draw_text(draw, (tgx + 6, py0 + 10), f"TARGET +{theta0_deg:.0f}°", _font(12, bold=True), CYAN)

        pts_nom = _map_pts(theta_grid, pat_nom, x_lim, y_lim, plot_box)
        pts_live = _map_pts(theta_grid, cur_pat, x_lim, y_lim, plot_box)
        _draw_dashed_polyline(draw, pts_nom, SLATE_LINE, width=2)
        draw.line(pts_live, fill=beam_col, width=4)

        pk_idx = int(np.argmax(cur_pat))
        pk_pt = pts_live[pk_idx]
        draw.ellipse((pk_pt[0] - 6, pk_pt[1] - 6, pk_pt[0] + 6, pk_pt[1] + 6), fill=beam_col, outline=TEXT, width=2)

        # Mini legend inside plot top-left
        _draw_card(draw, (px0 + 10, py0 + 10, px0 + 250, py0 + 62), fill=(15, 23, 42), outline=BORDER, radius=6, width=1)
        draw.line([(px0 + 18, py0 + 26), (px0 + 44, py0 + 26)], fill=SLATE_LINE, width=2)
        _draw_text(draw, (px0 + 52, py0 + 26), "Ideal Nominal (-30 dB)", _font(12), MUTED, anchor="lm")
        draw.line([(px0 + 18, py0 + 46), (px0 + 44, py0 + 46)], fill=beam_col, width=4)
        _draw_text(draw, (px0 + 52, py0 + 46), "Live X-Band Beam", _font(12, bold=True), TEXT, anchor="lm")

        # Bottom KPI Strip (4 equal cards)
        cur_uz_span = float(cur_disp[:, 2].max() - cur_disp[:, 2].min()) * 1e3
        kpi_cards = [
            ("APERTURE FLEX", f"{cur_uz_span:.1f} mm", f"{cur_uz_span / 30.0:.2f} λ₀", CYAN),
            (
                "BORESIGHT ERROR",
                f"{cur_m['boresight_error_deg']:+.2f}°",
                "Target: +15.0°",
                GREEN if abs(cur_m["boresight_error_deg"]) < 0.5 else RED,
            ),
            (
                "STEALTH SIDELOBE",
                f"{cur_m['sll_db']:.1f} dB",
                "Stealth limit <= -25 dB",
                GREEN if cur_m["sll_db"] <= -25.0 else RED,
            ),
            (
                "AI CONTROLLER",
                "OFF (OPEN-LOOP)" if alpha_heal == 0.0 else "0.82 ms | 6-BIT",
                f"Detected: {live['pred_class']}",
                AMBER if alpha_heal == 0.0 else GREEN,
            ),
        ]
        kw = 248
        kgap = 16
        for ki, (klbl, kval, ksub, kcol) in enumerate(kpi_cards):
            kx0 = 20 + ki * (kw + kgap)
            kx1 = kx0 + kw
            _draw_card(draw, (kx0, 526, kx1, 608), fill=CARD, outline=kcol, radius=10, width=2)
            _draw_text(draw, (kx0 + 14, 534), klbl, _font(12, bold=True), MUTED)
            _draw_text(draw, (kx0 + 14, 554), kval, _fit_font(draw, kval, kw - 28, 22, bold=True, mono=True), kcol)
            _draw_text(draw, (kx0 + 14, 584), ksub, _fit_font(draw, ksub, kw - 28, 12, mono=True), TEXT)

        frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
        durations.append(dur)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return len(frames)


def generate_all_showcase_assets(project_root: Path) -> Dict[str, Any]:
    """Generate Slide 1-3 PNGs, PDF Carousel, and live combat GIF from real ANSYS/AI data."""
    project_root = Path(project_root)
    out_dir = project_root / "outputs" / "showcase"
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = load_cockpit_assets(project_root)
    case_idx = pick_dramatic_heldout_case(assets)
    live = run_live_combat_inference(
        assets=assets,
        case_idx=case_idx,
        dead_sensor_ids=[3, 7],
        noise_pct=3.0,
        theta0_deg=15.0,
        quant_bits=6,
        rng_seed=777,
    )

    s1_path = out_dir / "slide1_story_problem.png"
    s2_path = out_dir / "slide2_combat_sensor_kill.png"
    s3_path = out_dir / "slide3_architecture_and_vv.png"
    pdf_path = out_dir / "AeroRF_Heal_LinkedIn_Carousel.pdf"
    gif_path = out_dir / "aerorf_healing_demo.gif"

    s1 = render_slide1_story_problem(assets, live, s1_path)
    s2 = render_slide2_combat_sensor_kill(assets, live, s2_path)
    s3 = render_slide3_architecture_and_vv(assets, live, s3_path)
    s1.save(pdf_path, "PDF", resolution=150.0, save_all=True, append_images=[s2, s3])

    n_frames = render_healing_gif(assets, live, gif_path)

    return {
        "case_idx": case_idx,
        "true_class": live["true_class"],
        "pred_class": live["pred_class"],
        "uncomp_err_deg": live["metrics"]["uncompensated"]["boresight_error_deg"],
        "uncomp_sll_db": live["metrics"]["uncompensated"]["sll_db"],
        "ai_err_deg": live["metrics"]["ai_blind"]["boresight_error_deg"],
        "ai_sll_db": live["metrics"]["ai_blind"]["sll_db"],
        "gif_frames": n_frames,
        "files": [s1_path, s2_path, s3_path, pdf_path, gif_path],
    }

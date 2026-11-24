"""Configuration dataclasses for AeroRF-Heal (strict SI units: m, N, Pa, kg, degC, rad)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class WingboxConfig:
    """Parametric multi-cell composite UAV wingbox (AS4/3501-6 Carbon/Epoxy)."""

    span: float = 1.20
    chord: float = 0.45
    height: float = 0.06
    y_fspar: float = 0.08
    y_rspar: float = 0.37
    rib_x: Tuple[float, ...] = (0.0, 0.30, 0.60, 0.90, 1.20)

    # AS4/3501-6 Carbon/Epoxy orthotropic properties (SI)
    e1: float = 142.0e9
    e2: float = 10.3e9
    g12: float = 7.2e9
    g13: float = 7.2e9
    g23: float = 3.7e9
    nu12: float = 0.27
    rho: float = 1580.0
    alpha1: float = -0.9e-6
    alpha2: float = 27.0e-6

    # Symmetric 8-ply quasi-isotropic layup [0/+45/-45/90]_s
    ply_angles: Tuple[float, ...] = (0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0)
    skin_ply_thickness: float = 0.00020      # 8 * 0.20 mm = 1.6 mm
    internal_ply_thickness: float = 0.00025  # 8 * 0.25 mm = 2.0 mm

    # Global target element edge size (keeps node count ~4k-8k, well inside Student cap)
    esize: float = 0.025

    @property
    def z_top(self) -> float:
        return 0.5 * self.height

    @property
    def z_bot(self) -> float:
        return -0.5 * self.height

    @property
    def spar_y_stations(self) -> Tuple[float, ...]:
        return (0.0, self.y_fspar, self.y_rspar, self.chord)


@dataclass(frozen=True)
class ArrayConfig:
    """X-Band 16x16 conformal phased-array radar embedded on upper wing skin."""

    freq_hz: float = 10.0e9
    c0: float = 299792458.0
    nx: int = 16
    ny: int = 16
    x_center: float = 0.65
    y_center: float = 0.225
    z_surface: float = 0.03
    q_elem: float = 1.4
    sll_target_db: float = 30.0
    nbar: int = 4
    quant_bits: int = 6

    @property
    def lambda0(self) -> float:
        return self.c0 / self.freq_hz

    @property
    def k0(self) -> float:
        return 2.0 * math.pi / self.lambda0

    @property
    def dx(self) -> float:
        return 0.5 * self.lambda0

    @property
    def dy(self) -> float:
        return 0.5 * self.lambda0

    @property
    def num_elements(self) -> int:
        return self.nx * self.ny

    def nominal_element_positions(self) -> np.ndarray:
        """Return (N, 3) nominal (x, y, z) coordinates of the 2D patch array."""
        xs = self.x_center + (np.arange(self.nx) - 0.5 * (self.nx - 1)) * self.dx
        ys = self.y_center + (np.arange(self.ny) - 0.5 * (self.ny - 1)) * self.dy
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        grid_z = np.full_like(grid_x, self.z_surface)
        return np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

    def nominal_normals(self) -> np.ndarray:
        """Return (N, 3) nominal broadside unit normals (0, 0, +1)."""
        normals = np.zeros((self.num_elements, 3), dtype=np.float64)
        normals[:, 2] = 1.0
        return normals


@dataclass(frozen=True)
class FBGSensorConfig:
    """12 sparse upper-skin Fiber Bragg Grating (FBG) strain sensor locations."""

    sensor_xy: Tuple[Tuple[float, float], ...] = (
        (0.15, 0.085),
        (0.15, 0.225),
        (0.15, 0.365),
        (0.42, 0.085),
        (0.42, 0.225),
        (0.42, 0.365),
        (0.65, 0.085),
        (0.65, 0.365),
        (0.88, 0.085),
        (0.88, 0.225),
        (0.88, 0.365),
        (1.10, 0.225),
    )

    @property
    def num_sensors(self) -> int:
        return len(self.sensor_xy)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.sensor_xy, dtype=np.float64)


@dataclass(frozen=True)
class WingLoadCase:
    """Coupled aerodynamic, torsional, thermal, and optional battle-damage load case."""

    name: str = "5G_maneuver_thermal"
    p0_pa: float = 35000.0
    torque_grad: float = 0.65
    delta_t_upper_c: float = 45.0
    delta_t_lower_c: float = 5.0
    t_ref_c: float = 20.0
    damage_zone: Optional[str] = None
    damage_factor: float = 0.15

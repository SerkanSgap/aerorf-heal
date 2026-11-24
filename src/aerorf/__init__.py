"""AeroRF-Heal: Self-Healing Conformal AESA Radar Digital Twin."""

from .ansys_runner import (
    MapdlRunResult,
    find_mapdl_executable,
    generate_cantilever_vv_apdl,
    generate_wingbox_apdl,
    interpolate_to_array,
    load_fbg_sensors,
    load_upper_skin_nodes,
    run_mapdl_batch,
)
from .config import (
    ArrayConfig,
    FBGSensorConfig,
    WingLoadCase,
    WingboxConfig,
)
from .rf_array import (
    compute_pattern_cut,
    compute_pattern_uv_grid,
    compute_steering_weights,
    evaluate_conformal_scenario,
    evaluate_pattern_metrics,
    make_2d_taylor_taper,
    quantize_phase,
    rotate_normals_euler,
)

__all__ = [
    "ArrayConfig",
    "FBGSensorConfig",
    "MapdlRunResult",
    "WingLoadCase",
    "WingboxConfig",
    "compute_pattern_cut",
    "compute_pattern_uv_grid",
    "compute_steering_weights",
    "evaluate_conformal_scenario",
    "evaluate_pattern_metrics",
    "find_mapdl_executable",
    "generate_cantilever_vv_apdl",
    "generate_wingbox_apdl",
    "interpolate_to_array",
    "load_fbg_sensors",
    "load_upper_skin_nodes",
    "make_2d_taylor_taper",
    "quantize_phase",
    "rotate_normals_euler",
    "run_mapdl_batch",
]

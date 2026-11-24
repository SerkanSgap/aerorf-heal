"""Headless ANSYS MAPDL 2026 R1 batch runner, V&V benchmark, and composite wingbox generator."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay

from .config import ArrayConfig, FBGSensorConfig, WingLoadCase, WingboxConfig

DAMAGE_CLASSES: Tuple[str, ...] = (
    "intact",
    "front_spar_bay2",
    "rear_spar_bay2",
    "upper_skin_bay2",
)


@dataclass(frozen=True)
class DOELoadSample:
    """Single parameterized flight + thermal + damage scenario for batch ANSYS DOE."""

    p0_pa: float
    torque_grad: float
    p_gust_pa: float
    x_gust_m: float
    dt_upper_c: float
    dt_lower_c: float
    dt_radar_c: float
    damage_factor: float = 1.0


@dataclass
class MapdlRunResult:
    """Structured summary of a headless ANSYS MAPDL batch execution."""

    returncode: int
    elapsed_sec: float
    work_dir: Path
    out_path: Path
    node_count: int
    element_count: int
    errors: List[str]
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and len(self.errors) == 0


def find_mapdl_executable() -> Path:
    """Auto-detect ANSYS MAPDL executable from env vars or standard install paths."""
    env_candidates = []
    for env_key in ("ANSYS261_DIR", "ANSYS252_DIR", "ANSYS251_DIR", "ANSYS242_DIR"):
        val = os.environ.get(env_key)
        if val:
            env_candidates.append(Path(val) / "bin" / "winx64" / "MAPDL.exe")
            env_candidates.append(Path(val) / "bin" / "winx64" / "ANSYS.exe")
    for root_key in ("AWP_ROOT261", "AWP_ROOT252", "AWP_ROOT251"):
        val = os.environ.get(root_key)
        if val:
            env_candidates.append(Path(val) / "ANSYS" / "bin" / "winx64" / "MAPDL.exe")

    for cand in env_candidates:
        if cand.is_file():
            return cand

    glob_patterns = [
        r"C:\Program Files\ANSYS Inc\ANSYS Student\v*\ANSYS\bin\winx64\MAPDL.exe",
        r"C:\Program Files\ANSYS Inc\v*\ANSYS\bin\winx64\MAPDL.exe",
    ]
    for pattern in glob_patterns:
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            return Path(matches[0])

    raise FileNotFoundError(
        "Could not locate MAPDL.exe. Check ANSYS261_DIR / AWP_ROOT261 environment variables."
    )


def _parse_mapdl_output(out_path: Path) -> Tuple[int, int, List[str], List[str]]:
    """Extract node count, element count, errors, and warnings from mapdl.out."""
    if not out_path.is_file():
        return 0, 0, [f"Missing MAPDL output file: {out_path}"], []

    text = out_path.read_text(encoding="utf-8", errors="replace")
    node_count = 0
    elem_count = 0
    errors: List[str] = []
    warnings: List[str] = []

    m_nodes = re.findall(r"MAXIMUM NODE NUMBER\s*=\s*(\d+)", text)
    if m_nodes:
        node_count = int(m_nodes[-1])
    else:
        m_n2 = re.findall(r"NUMBER OF NODES\s*=\s*(\d+)", text)
        if m_n2:
            node_count = int(m_n2[-1])

    m_elems = re.findall(r"MAXIMUM ELEMENT NUMBER\s*=\s*(\d+)", text)
    if m_elems:
        elem_count = int(m_elems[-1])
    else:
        m_e2 = re.findall(r"NUMBER OF ELEMENTS\s*=\s*(\d+)", text)
        if m_e2:
            elem_count = int(m_e2[-1])

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "*** ERROR ***" in line:
            snippet = " | ".join(lines[i : min(i + 4, len(lines))]).strip()
            errors.append(snippet)
        elif "*** WARNING ***" in line:
            snippet = " | ".join(lines[i : min(i + 2, len(lines))]).strip()
            warnings.append(snippet)

    return node_count, elem_count, errors, warnings


def run_mapdl_batch(
    apdl_path: Path,
    work_dir: Path,
    nproc: int = 4,
    timeout: int = 180,
    jobname: str = "aerorf",
) -> MapdlRunResult:
    """Execute an APDL script headlessly with MAPDL.exe in batch mode."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    apdl_path = Path(apdl_path).resolve()
    out_path = work_dir / "mapdl.out"

    # Remove stale lock/output files if re-running in same directory
    for stale in (out_path, work_dir / f"{jobname}.lock"):
        if stale.exists():
            stale.unlink()

    exe = find_mapdl_executable()
    cmd = [
        str(exe),
        "-b",
        "-np",
        str(nproc),
        "-j",
        jobname,
        "-i",
        str(apdl_path),
        "-o",
        str(out_path),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0

    node_count, elem_count, errors, warnings = _parse_mapdl_output(out_path)
    if proc.returncode != 0 and not errors:
        errors.append(f"MAPDL exited with return code {proc.returncode}: {proc.stderr.strip()}")

    return MapdlRunResult(
        returncode=proc.returncode,
        elapsed_sec=elapsed,
        work_dir=work_dir,
        out_path=out_path,
        node_count=node_count,
        element_count=elem_count,
        errors=errors,
        warnings=warnings,
    )


def generate_cantilever_vv_apdl(
    apdl_path: Path,
    output_csv_name: str = "cantilever_vv.csv",
) -> Path:
    """Generate slender aluminum cantilever plate V&V benchmark APDL script.

    Geometry: L = 1.0 m, b = 0.05 m, t = 0.01 m, E = 70e9 Pa, nu = 0.33.
    Tip force F = 10.0 N distributed across tip edge x = L (SHELL181, esize=0.025).
    Euler-Bernoulli reference: delta = F*L^3 / (3*E*I) = 0.01142857 m (11.429 mm).
    """
    apdl_path = Path(apdl_path)
    apdl_path.parent.mkdir(parents=True, exist_ok=True)
    stem = Path(output_csv_name).stem

    lines = [
        "/FINISH",
        "/CLEAR,NOSTART",
        "/PREP7",
        "! Cantilever Plate V&V Benchmark (SI: m, N, Pa)",
        "ET,1,SHELL181",
        "SECTYPE,1,SHELL,,AL_PLATE",
        "SECDATA,0.01,1,0.0,3",
        "SECOFFSET,MID",
        "MP,EX,1,70.0E9",
        "MP,PRXY,1,0.33",
        "MP,DENS,1,2700.0",
        "K,1,0.0,0.0,0.0",
        "K,2,1.0,0.0,0.0",
        "K,3,1.0,0.05,0.0",
        "K,4,0.0,0.05,0.0",
        "A,1,2,3,4",
        "ESIZE,0.025",
        "AMESH,1",
        "NSEL,S,LOC,X,0.0",
        "D,ALL,ALL,0.0",
        "NSEL,S,LOC,X,1.0",
        "*GET,NTIP,NODE,0,COUNT",
        "FZ_NODE = 10.0/NTIP",
        "F,ALL,FZ,FZ_NODE",
        "ALLSEL,ALL",
        "FINISH",
        "/SOLU",
        "ANTYPE,STATIC",
        "SOLVE",
        "FINISH",
        "/POST1",
        "SET,LAST",
        "ALLSEL,ALL",
        "*GET,NCNT,NODE,0,COUNT",
        "*GET,ECNT,ELEM,0,COUNT",
        "NSEL,S,LOC,X,1.0",
        "NSORT,U,Z,0,1",
        "*GET,UZ_MAX,SORT,,MAX",
        "ALLSEL,ALL",
        f"*CFOPEN,{stem},csv",
        "*VWRITE,NCNT,ECNT,UZ_MAX",
        "(F10.0,',',F10.0,',',E16.8)",
        "*CFCLOS",
        "FINISH",
        "/EXIT,NOSAVE",
    ]
    apdl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return apdl_path


def _build_wingbox_prep_lines(
    wing_cfg: WingboxConfig,
    damage_zone: Optional[str],
    damage_factor: float,
    label: str,
) -> List[str]:
    """Build shared /PREP7 composite wingbox materials, sections, geometry, mesh, and root BC."""
    rib_x = wing_cfg.rib_x
    spar_y = wing_cfg.spar_y_stations
    z_top = wing_cfg.z_top
    z_bot = wing_cfg.z_bot

    lines: List[str] = [
        "/FINISH",
        "/CLEAR,NOSTART",
        "/PREP7",
        f"! AeroRF-Heal Composite Wingbox: {label}",
        "ET,1,SHELL181",
        "KEYOPT,1,8,2",
        "! Mat 1: AS4/3501-6 Carbon/Epoxy (SI)",
        f"MP,EX,1,{wing_cfg.e1:.6E}",
        f"MP,EY,1,{wing_cfg.e2:.6E}",
        f"MP,EZ,1,{wing_cfg.e2:.6E}",
        f"MP,PRXY,1,{wing_cfg.nu12:.6f}",
        f"MP,PRYZ,1,{wing_cfg.nu12:.6f}",
        f"MP,PRXZ,1,{wing_cfg.nu12:.6f}",
        f"MP,GXY,1,{wing_cfg.g12:.6E}",
        f"MP,GXZ,1,{wing_cfg.g13:.6E}",
        f"MP,GYZ,1,{wing_cfg.g23:.6E}",
        f"MP,DENS,1,{wing_cfg.rho:.2f}",
        f"MP,ALPX,1,{wing_cfg.alpha1:.6E}",
        f"MP,ALPY,1,{wing_cfg.alpha2:.6E}",
        f"MP,ALPZ,1,{wing_cfg.alpha2:.6E}",
        "! Mat 2: Battle-Damage Knockdown Material",
        f"MP,EX,2,{wing_cfg.e1 * damage_factor:.6E}",
        f"MP,EY,2,{wing_cfg.e2 * damage_factor:.6E}",
        f"MP,EZ,2,{wing_cfg.e2 * damage_factor:.6E}",
        f"MP,PRXY,2,{wing_cfg.nu12:.6f}",
        f"MP,PRYZ,2,{wing_cfg.nu12:.6f}",
        f"MP,PRXZ,2,{wing_cfg.nu12:.6f}",
        f"MP,GXY,2,{wing_cfg.g12 * damage_factor:.6E}",
        f"MP,GXZ,2,{wing_cfg.g13 * damage_factor:.6E}",
        f"MP,GYZ,2,{wing_cfg.g23 * damage_factor:.6E}",
        f"MP,DENS,2,{wing_cfg.rho:.2f}",
        f"MP,ALPX,2,{wing_cfg.alpha1:.6E}",
        f"MP,ALPY,2,{wing_cfg.alpha2:.6E}",
        f"MP,ALPZ,2,{wing_cfg.alpha2:.6E}",
    ]

    lines.append("SECTYPE,1,SHELL,,SKIN_INTACT")
    for ang in wing_cfg.ply_angles:
        lines.append(f"SECDATA,{wing_cfg.skin_ply_thickness:.6f},1,{ang:.1f},3")
    lines.append("SECOFFSET,MID")

    lines.append("SECTYPE,2,SHELL,,INT_INTACT")
    for ang in wing_cfg.ply_angles:
        lines.append(f"SECDATA,{wing_cfg.internal_ply_thickness:.6f},1,{ang:.1f},3")
    lines.append("SECOFFSET,MID")

    lines.append("SECTYPE,3,SHELL,,SKIN_DMG")
    for ang in wing_cfg.ply_angles:
        lines.append(f"SECDATA,{wing_cfg.skin_ply_thickness:.6f},2,{ang:.1f},3")
    lines.append("SECOFFSET,MID")

    lines.append("SECTYPE,4,SHELL,,INT_DMG")
    for ang in wing_cfg.ply_angles:
        lines.append(f"SECDATA,{wing_cfg.internal_ply_thickness:.6f},2,{ang:.1f},3")
    lines.append("SECOFFSET,MID")

    kp_top: Dict[Tuple[int, int], int] = {}
    kp_bot: Dict[Tuple[int, int], int] = {}
    kid = 1
    for ix, x in enumerate(rib_x):
        for iy, y in enumerate(spar_y):
            lines.append(f"K,{kid},{x:.5f},{y:.5f},{z_top:.5f}")
            kp_top[(ix, iy)] = kid
            kid += 1
            lines.append(f"K,{kid},{x:.5f},{y:.5f},{z_bot:.5f}")
            kp_bot[(ix, iy)] = kid
            kid += 1

    skin_area_ids: List[int] = []
    internal_area_ids: List[int] = []
    fspar_areas: List[Tuple[int, int]] = []
    rspar_areas: List[Tuple[int, int]] = []
    upper_bay_areas: List[Tuple[int, int]] = []
    aid = 1

    for ix in range(len(rib_x) - 1):
        for iy in range(len(spar_y) - 1):
            k1 = kp_top[(ix, iy)]
            k2 = kp_top[(ix + 1, iy)]
            k3 = kp_top[(ix + 1, iy + 1)]
            k4 = kp_top[(ix, iy + 1)]
            lines.append(f"A,{k1},{k2},{k3},{k4}")
            skin_area_ids.append(aid)
            if iy == 1:
                upper_bay_areas.append((ix, aid))
            aid += 1

            b1 = kp_bot[(ix, iy)]
            b2 = kp_bot[(ix + 1, iy)]
            b3 = kp_bot[(ix + 1, iy + 1)]
            b4 = kp_bot[(ix, iy + 1)]
            lines.append(f"A,{b1},{b2},{b3},{b4}")
            skin_area_ids.append(aid)
            aid += 1

    for ix in range(len(rib_x) - 1):
        for iy_spar, store in ((1, fspar_areas), (2, rspar_areas)):
            k1 = kp_bot[(ix, iy_spar)]
            k2 = kp_bot[(ix + 1, iy_spar)]
            k3 = kp_top[(ix + 1, iy_spar)]
            k4 = kp_top[(ix, iy_spar)]
            lines.append(f"A,{k1},{k2},{k3},{k4}")
            internal_area_ids.append(aid)
            store.append((ix, aid))
            aid += 1

    for ix in range(1, len(rib_x)):
        for iy in range(len(spar_y) - 1):
            k1 = kp_bot[(ix, iy)]
            k2 = kp_bot[(ix, iy + 1)]
            k3 = kp_top[(ix, iy + 1)]
            k4 = kp_top[(ix, iy)]
            lines.append(f"A,{k1},{k2},{k3},{k4}")
            internal_area_ids.append(aid)
            aid += 1

    lines.append("NUMMRG,KP,1.0E-5")

    for a in skin_area_ids:
        lines.extend([f"ASEL,S,AREA,,{a}", "AATT,1,,1,0,1"])
    for a in internal_area_ids:
        lines.extend([f"ASEL,S,AREA,,{a}", "AATT,1,,1,0,2"])

    if damage_zone == "front_spar_bay2":
        dmg_a = dict(fspar_areas).get(1)
        if dmg_a is not None:
            lines.extend([f"ASEL,S,AREA,,{dmg_a}", "AATT,2,,1,0,4"])
    elif damage_zone == "rear_spar_bay2":
        dmg_a = dict(rspar_areas).get(1)
        if dmg_a is not None:
            lines.extend([f"ASEL,S,AREA,,{dmg_a}", "AATT,2,,1,0,4"])
    elif damage_zone == "upper_skin_bay2":
        dmg_a = dict(upper_bay_areas).get(1)
        if dmg_a is not None:
            lines.extend([f"ASEL,S,AREA,,{dmg_a}", "AATT,2,,1,0,3"])

    lines.extend([
        "ALLSEL,ALL",
        f"ESIZE,{wing_cfg.esize:.4f}",
        "AMESH,ALL",
        "! Root cantilever constraint at x = 0",
        "NSEL,S,LOC,X,-1.0E-4,1.0E-4",
        "D,ALL,ALL,0.0",
        "ALLSEL,ALL",
    ])
    return lines


def generate_wingbox_apdl(
    apdl_path: Path,
    wing_cfg: WingboxConfig,
    fbg_cfg: FBGSensorConfig,
    load_case: WingLoadCase,
    upper_skin_csv_name: str = "upper_skin_nodes.csv",
    fbg_csv_name: str = "fbg_sensors.csv",
) -> Path:
    """Generate parametric multi-cell AS4/3501-6 composite UAV wingbox APDL script."""
    apdl_path = Path(apdl_path)
    apdl_path.parent.mkdir(parents=True, exist_ok=True)

    upper_stem = Path(upper_skin_csv_name).stem
    fbg_stem = Path(fbg_csv_name).stem
    z_top = wing_cfg.z_top
    z_bot = wing_cfg.z_bot
    n_plies = len(wing_cfg.ply_angles)

    lines = _build_wingbox_prep_lines(
        wing_cfg,
        load_case.damage_zone,
        load_case.damage_factor,
        load_case.name,
    )

    lines.extend([
        "! Distributed aerodynamic lift + torsional pressure on upper skin",
        f"ESEL,S,CENT,Z,{z_top - 1e-4:.5f},{z_top + 1e-4:.5f}",
        "*GET,ECNT_U,ELEM,0,COUNT",
        "EID = 0",
        "*DO,II,1,ECNT_U",
        "  EID = ELNEXT(EID)",
        "  *GET,XC,ELEM,EID,CENT,X",
        "  *GET,YC,ELEM,EID,CENT,Y",
        f"  XR = XC/{wing_cfg.span:.4f}",
        "  SQVAL = 1.0 - XR*XR",
        "  *IF,SQVAL,LT,0.0,THEN",
        "    SQVAL = 0.0",
        "  *ENDIF",
        f"  TTERM = 1.0 + {load_case.torque_grad:.4f}*((YC-{0.5 * wing_cfg.chord:.4f})/{wing_cfg.chord:.4f})",
        f"  PVAL = {load_case.p0_pa:.2f}*SQRT(SQVAL)*TTERM",
        "  SFE,EID,1,PRES,1,PVAL",
        "*ENDDO",
        "ALLSEL,ALL",
        "! Coupled smooth thermal gradient through thickness",
        f"TREF,{load_case.t_ref_c:.2f}",
        f"TLOW = {load_case.t_ref_c + load_case.delta_t_lower_c:.2f}",
        f"TUPP = {load_case.t_ref_c + load_case.delta_t_upper_c:.2f}",
        "*GET,NCNT_ALL,NODE,0,COUNT",
        "NID = 0",
        "*DO,II,1,NCNT_ALL",
        "  NID = NDNEXT(NID)",
        "  *GET,ZN,NODE,NID,LOC,Z",
        f"  ZFRAC = (ZN - ({z_bot:.5f}))/({wing_cfg.height:.5f})",
        "  TN = TLOW + (TUPP - TLOW)*ZFRAC",
        "  BF,NID,TEMP,TN",
        "*ENDDO",
        "ALLSEL,ALL",
        "FINISH",
        "/SOLU",
        "ANTYPE,STATIC",
        "NLGEOM,OFF",
        "SOLVE",
        "FINISH",
        "/POST1",
        "SET,LAST",
        "! Export all upper-skin nodal displacements and rotations",
        f"NSEL,S,LOC,Z,{z_top - 1e-4:.5f},{z_top + 1e-4:.5f}",
        "*GET,NCNT,NODE,0,COUNT",
        "*DIM,WID,ARRAY,NCNT",
        "*DIM,WX,ARRAY,NCNT",
        "*DIM,WY,ARRAY,NCNT",
        "*DIM,WZ,ARRAY,NCNT",
        "*DIM,WUX,ARRAY,NCNT",
        "*DIM,WUY,ARRAY,NCNT",
        "*DIM,WUZ,ARRAY,NCNT",
        "*DIM,WRX,ARRAY,NCNT",
        "*DIM,WRY,ARRAY,NCNT",
        "*DIM,WRZ,ARRAY,NCNT",
        "NN = 0",
        "*DO,II,1,NCNT",
        "  NN = NDNEXT(NN)",
        "  WID(II) = NN",
        "  *GET,WX(II),NODE,NN,LOC,X",
        "  *GET,WY(II),NODE,NN,LOC,Y",
        "  *GET,WZ(II),NODE,NN,LOC,Z",
        "  *GET,WUX(II),NODE,NN,U,X",
        "  *GET,WUY(II),NODE,NN,U,Y",
        "  *GET,WUZ(II),NODE,NN,U,Z",
        "  *GET,WRX(II),NODE,NN,ROT,X",
        "  *GET,WRY(II),NODE,NN,ROT,Y",
        "  *GET,WRZ(II),NODE,NN,ROT,Z",
        "*ENDDO",
        f"*CFOPEN,{upper_stem},csv",
        "*VWRITE,WID(1),WX(1),WY(1),WZ(1),WUX(1),WUY(1),WUZ(1),WRX(1),WRY(1),WRZ(1)",
        "(F10.0,9(',',E15.7))",
        "*CFCLOS",
        "! Extract pure upper-skin top-ply strains at the 12 FBG sensors",
        f"ESEL,S,CENT,Z,{z_top - 1e-4:.5f},{z_top + 1e-4:.5f}",
        "NSLE,S",
        f"LAYER,{n_plies}",
        "SHELL,TOP",
        "RSYS,0",
        "AVPRIN,0",
        "/OUT,tmp_epto,txt",
        "PRNSOL,EPTO,COMP",
        "/OUT",
        f"*DIM,FID,ARRAY,{fbg_cfg.num_sensors}",
        f"*DIM,FXT,ARRAY,{fbg_cfg.num_sensors}",
        f"*DIM,FYT,ARRAY,{fbg_cfg.num_sensors}",
        f"*DIM,FNX,ARRAY,{fbg_cfg.num_sensors}",
        f"*DIM,FNY,ARRAY,{fbg_cfg.num_sensors}",
        f"*DIM,FEX,ARRAY,{fbg_cfg.num_sensors}",
        f"*DIM,FEY,ARRAY,{fbg_cfg.num_sensors}",
        f"*DIM,FEXY,ARRAY,{fbg_cfg.num_sensors}",
    ])

    for idx, (sx, sy) in enumerate(fbg_cfg.sensor_xy, start=1):
        lines.extend([
            f"FID({idx}) = {idx}",
            f"FXT({idx}) = {sx:.5f}",
            f"FYT({idx}) = {sy:.5f}",
            f"SNID = NODE({sx:.5f},{sy:.5f},{z_top:.5f})",
            f"*GET,FNX({idx}),NODE,SNID,LOC,X",
            f"*GET,FNY({idx}),NODE,SNID,LOC,Y",
            f"*GET,FEX({idx}),NODE,SNID,EPTO,X",
            f"*GET,FEY({idx}),NODE,SNID,EPTO,Y",
            f"*GET,FEXY({idx}),NODE,SNID,EPTO,XY",
        ])

    lines.extend([
        f"*CFOPEN,{fbg_stem},csv",
        "*VWRITE,FID(1),FXT(1),FYT(1),FNX(1),FNY(1),FEX(1),FEY(1),FEXY(1)",
        "(F6.0,7(',',E15.7))",
        "*CFCLOS",
        "ALLSEL,ALL",
        "FINISH",
        "/EXIT,NOSAVE",
    ])

    apdl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return apdl_path


def generate_batch_doe_apdl(
    apdl_path: Path,
    wing_cfg: WingboxConfig,
    fbg_cfg: FBGSensorConfig,
    damage_zone: str,
    samples: List[DOELoadSample],
    t_ref_c: float = 20.0,
) -> Path:
    """Generate single-session multi-loadcase ANSYS MAPDL DOE script for one topology."""
    apdl_path = Path(apdl_path)
    work_dir = apdl_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    ncases = len(samples)
    if ncases == 0:
        raise ValueError("samples must be non-empty")

    z_top = wing_cfg.z_top
    z_bot = wing_cfg.z_bot
    n_plies = len(wing_cfg.ply_angles)
    n_fbg = fbg_cfg.num_sensors

    # Write clean sub-macro for *VWRITE inside *DO loop (ANSYS-documented pattern)
    export_macro = work_dir / "export_case.mac"
    export_macro.write_text(
        "\n".join([
            "*CFOPEN,batch_aperture,csv,,APPEND",
            "*VWRITE,AIC(1),AID(1),AUX(1),AUY(1),AUZ(1),ARX(1),ARY(1),ARZ(1)",
            "(F6.0,',',F8.0,6(',',E15.7))",
            "*CFCLOS",
            "*CFOPEN,batch_fbg,csv,,APPEND",
            "*VWRITE,FIC(1),FID(1),FEX(1),FEY(1),FXY(1)",
            "(F6.0,',',F6.0,3(',',E15.7))",
            "*CFCLOS",
        ]) + "\n",
        encoding="utf-8",
    )

    lines = _build_wingbox_prep_lines(
        wing_cfg,
        damage_zone,
        samples[0].damage_factor,
        f"DOE_{damage_zone}",
    )

    # Pre-identify upper-skin aperture nodes (x in [0.50, 0.80], y in [0.075, 0.375])
    lines.extend([
        "! Pre-select aperture patch nodes once and export coordinates",
        f"NSEL,S,LOC,Z,{z_top - 1e-4:.5f},{z_top + 1e-4:.5f}",
        "NSEL,R,LOC,X,0.4999,0.8001",
        "NSEL,R,LOC,Y,0.0749,0.3751",
        "*GET,NAP,NODE,0,COUNT",
        "*DIM,AID,ARRAY,NAP",
        "*DIM,AXC,ARRAY,NAP",
        "*DIM,AYC,ARRAY,NAP",
        "*DIM,AIC,ARRAY,NAP",
        "*DIM,AUX,ARRAY,NAP",
        "*DIM,AUY,ARRAY,NAP",
        "*DIM,AUZ,ARRAY,NAP",
        "*DIM,ARX,ARRAY,NAP",
        "*DIM,ARY,ARRAY,NAP",
        "*DIM,ARZ,ARRAY,NAP",
        "NN = 0",
        "*DO,JJ,1,NAP",
        "  NN = NDNEXT(NN)",
        "  AID(JJ) = NN",
        "  *GET,AXC(JJ),NODE,NN,LOC,X",
        "  *GET,AYC(JJ),NODE,NN,LOC,Y",
        "*ENDDO",
        "*CFOPEN,aperture_mesh_xy,csv",
        "*VWRITE,AID(1),AXC(1),AYC(1)",
        "(F8.0,2(',',E15.7))",
        "*CFCLOS",
        "! Pre-identify 12 FBG upper-skin node IDs once",
        f"ESEL,S,CENT,Z,{z_top - 1e-4:.5f},{z_top + 1e-4:.5f}",
        "NSLE,S",
        f"*DIM,FID,ARRAY,{n_fbg}",
        f"*DIM,FNID,ARRAY,{n_fbg}",
        f"*DIM,FXN,ARRAY,{n_fbg}",
        f"*DIM,FYN,ARRAY,{n_fbg}",
        f"*DIM,FIC,ARRAY,{n_fbg}",
        f"*DIM,FEX,ARRAY,{n_fbg}",
        f"*DIM,FEY,ARRAY,{n_fbg}",
        f"*DIM,FXY,ARRAY,{n_fbg}",
    ])

    for idx, (sx, sy) in enumerate(fbg_cfg.sensor_xy, start=1):
        lines.extend([
            f"FID({idx}) = {idx}",
            f"FNID({idx}) = NODE({sx:.5f},{sy:.5f},{z_top:.5f})",
            f"*GET,FXN({idx}),NODE,FNID({idx}),LOC,X",
            f"*GET,FYN({idx}),NODE,FNID({idx}),LOC,Y",
        ])

    lines.extend([
        "*CFOPEN,fbg_mesh_xy,csv",
        "*VWRITE,FID(1),FNID(1),FXN(1),FYN(1)",
        "(F6.0,',',F8.0,2(',',E15.7))",
        "*CFCLOS",
        "ALLSEL,ALL",
        f"TREF,{t_ref_c:.2f}",
        f"NCASES = {ncases}",
        "*DIM,C_P0,ARRAY,NCASES",
        "*DIM,C_TG,ARRAY,NCASES",
        "*DIM,C_PG,ARRAY,NCASES",
        "*DIM,C_XG,ARRAY,NCASES",
        "*DIM,C_DTU,ARRAY,NCASES",
        "*DIM,C_DTL,ARRAY,NCASES",
        "*DIM,C_DTR,ARRAY,NCASES",
        "*DIM,C_DF,ARRAY,NCASES",
    ])

    for idx, s in enumerate(samples, start=1):
        lines.extend([
            f"C_P0({idx}) = {s.p0_pa:.2f}",
            f"C_TG({idx}) = {s.torque_grad:.4f}",
            f"C_PG({idx}) = {s.p_gust_pa:.2f}",
            f"C_XG({idx}) = {s.x_gust_m:.4f}",
            f"C_DTU({idx}) = {s.dt_upper_c:.2f}",
            f"C_DTL({idx}) = {s.dt_lower_c:.2f}",
            f"C_DTR({idx}) = {s.dt_radar_c:.2f}",
            f"C_DF({idx}) = {s.damage_factor:.4f}",
        ])

    # Pre-count upper elements and total nodes outside inner solve loop
    lines.extend([
        f"ESEL,S,CENT,Z,{z_top - 1e-4:.5f},{z_top + 1e-4:.5f}",
        "*GET,ECNT_U,ELEM,0,COUNT",
        "ALLSEL,ALL",
        "*GET,NCNT_ALL,NODE,0,COUNT",
        "FINISH",
        "! Multi-loadcase single-session DOE loop",
        "*DO,IC,1,NCASES",
        "  /PREP7",
        "  ALLSEL,ALL",
        "  *IF,IC,GT,1,THEN",
        "    SFEDELE,ALL,1,PRES",
        "    BFDELE,ALL,TEMP",
        "  *ENDIF",
        "  DFACT = C_DF(IC)",
        f"  V_EX2 = {wing_cfg.e1:.6E}*DFACT",
        f"  V_EY2 = {wing_cfg.e2:.6E}*DFACT",
        f"  V_G12 = {wing_cfg.g12:.6E}*DFACT",
        f"  V_G13 = {wing_cfg.g13:.6E}*DFACT",
        f"  V_G23 = {wing_cfg.g23:.6E}*DFACT",
        "  MP,EX,2,V_EX2",
        "  MP,EY,2,V_EY2",
        "  MP,EZ,2,V_EY2",
        "  MP,GXY,2,V_G12",
        "  MP,GXZ,2,V_G13",
        "  MP,GYZ,2,V_G23",
        f"  ESEL,S,CENT,Z,{z_top - 1e-4:.5f},{z_top + 1e-4:.5f}",
        "  /NOPR",
        "  EID = 0",
        "  *DO,II,1,ECNT_U",
        "    EID = ELNEXT(EID)",
        "    *GET,XC,ELEM,EID,CENT,X",
        "    *GET,YC,ELEM,EID,CENT,Y",
        f"    XR = XC/{wing_cfg.span:.4f}",
        "    SQV = 1.0 - XR*XR",
        "    *IF,SQV,LT,0.0,THEN",
        "      SQV = 0.0",
        "    *ENDIF",
        f"    TTERM = 1.0 + C_TG(IC)*((YC-{0.5 * wing_cfg.chord:.4f})/{wing_cfg.chord:.4f})",
        "    DXG = (XC - C_XG(IC))/0.20",
        "    GVAL = C_PG(IC)*EXP(-(DXG*DXG))",
        "    PVAL = C_P0(IC)*SQRT(SQV)*TTERM + GVAL",
        "    SFE,EID,1,PRES,1,PVAL",
        "  *ENDDO",
        "  ALLSEL,ALL",
        f"  TLOW = {t_ref_c:.2f} + C_DTL(IC)",
        f"  TUPP = {t_ref_c:.2f} + C_DTU(IC)",
        "  NID = 0",
        "  *DO,II,1,NCNT_ALL",
        "    NID = NDNEXT(NID)",
        "    *GET,XN,NODE,NID,LOC,X",
        "    *GET,YN,NODE,NID,LOC,Y",
        "    *GET,ZN,NODE,NID,LOC,Z",
        f"    ZFR = (ZN - ({z_bot:.5f}))/({wing_cfg.height:.5f})",
        "    DXR = XN - 0.6500",
        "    DYR = YN - 0.2250",
        "    DR2 = (DXR*DXR + DYR*DYR)/0.0250",
        "    TN = TLOW + (TUPP-TLOW)*ZFR + C_DTR(IC)*ZFR*EXP(-DR2)",
        "    BF,NID,TEMP,TN",
        "  *ENDDO",
        "  /GOPR",
        "  ALLSEL,ALL",
        "  FINISH",
        "  /SOLU",
        "  ANTYPE,STATIC",
        "  NLGEOM,OFF",
        "  SOLVE",
        "  FINISH",
        "  /POST1",
        "  SET,LAST",
        "  ALLSEL,ALL",
        "  /NOPR",
        "  *DO,JJ,1,NAP",
        "    NN = AID(JJ)",
        "    AIC(JJ) = IC",
        "    *GET,AUX(JJ),NODE,NN,U,X",
        "    *GET,AUY(JJ),NODE,NN,U,Y",
        "    *GET,AUZ(JJ),NODE,NN,U,Z",
        "    *GET,ARX(JJ),NODE,NN,ROT,X",
        "    *GET,ARY(JJ),NODE,NN,ROT,Y",
        "    *GET,ARZ(JJ),NODE,NN,ROT,Z",
        "  *ENDDO",
        f"  ESEL,S,CENT,Z,{z_top - 1e-4:.5f},{z_top + 1e-4:.5f}",
        "  NSLE,S",
        f"  LAYER,{n_plies}",
        "  SHELL,TOP",
        "  RSYS,0",
        "  AVPRIN,0",
        "  /OUT,tmp_epto,txt",
        "  PRNSOL,EPTO,COMP",
        "  /OUT",
        f"  *DO,KK,1,{n_fbg}",
        "    NN = FNID(KK)",
        "    FIC(KK) = IC",
        "    *GET,FEX(KK),NODE,NN,EPTO,X",
        "    *GET,FEY(KK),NODE,NN,EPTO,Y",
        "    *GET,FXY(KK),NODE,NN,EPTO,XY",
        "  *ENDDO",
        "  /GOPR",
        "  /INPUT,export_case,mac",
        "  ALLSEL,ALL",
        "  FINISH",
        "*ENDDO",
        "/EXIT,NOSAVE",
    ])

    apdl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return apdl_path


def run_doe_topology_batch(
    work_dir: Path,
    wing_cfg: WingboxConfig,
    array_cfg: ArrayConfig,
    fbg_cfg: FBGSensorConfig,
    damage_zone: str,
    samples: List[DOELoadSample],
    nproc: int = 4,
    timeout: int = 900,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MapdlRunResult]:
    """Run one single-session MAPDL DOE topology batch and return interpolated arrays.

    Returns (fbg_strains [N, 12, 3], array_disp [N, 256, 3], array_rot [N, 256, 3], run_result).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for csv_name in ("batch_aperture.csv", "batch_fbg.csv", "aperture_mesh_xy.csv", "fbg_mesh_xy.csv"):
        stale = work_dir / csv_name
        if stale.exists():
            stale.unlink()

    apdl_path = work_dir / f"doe_{damage_zone}.mac"
    generate_batch_doe_apdl(apdl_path, wing_cfg, fbg_cfg, damage_zone, samples)

    res = run_mapdl_batch(
        apdl_path,
        work_dir,
        nproc=nproc,
        timeout=timeout,
        jobname=f"doe_{damage_zone[:6]}",
    )
    if not res.ok:
        raise RuntimeError(f"MAPDL DOE batch '{damage_zone}' failed: {res.errors}")

    ap_xy_df = pd.read_csv(work_dir / "aperture_mesh_xy.csv", header=None, names=["node_id", "x", "y"])
    ap_xy_df["node_id"] = ap_xy_df["node_id"].astype(int)
    pts_xy = ap_xy_df[["x", "y"]].to_numpy(dtype=np.float64)
    nap = len(ap_xy_df)

    batch_ap_df = pd.read_csv(
        work_dir / "batch_aperture.csv",
        header=None,
        names=["case_id", "node_id", "ux", "uy", "uz", "rotx", "roty", "rotz"],
    )
    batch_fbg_df = pd.read_csv(
        work_dir / "batch_fbg.csv",
        header=None,
        names=["case_id", "sensor_id", "eps_x", "eps_y", "eps_xy"],
    )

    ncases = len(samples)
    n_fbg = fbg_cfg.num_sensors
    n_elem = array_cfg.num_elements
    assert len(batch_ap_df) == ncases * nap, f"Expected {ncases * nap} aperture rows, got {len(batch_ap_df)}"
    assert len(batch_fbg_df) == ncases * n_fbg, f"Expected {ncases * n_fbg} FBG rows, got {len(batch_fbg_df)}"

    # Precompute Delaunay triangulation once for fast per-case interpolation
    tri = Delaunay(pts_xy)
    target_xy = array_cfg.nominal_element_positions()[:, :2]
    near = NearestNDInterpolator(pts_xy, np.arange(nap))
    near_idx = near(target_xy).astype(int)

    ap_vals = batch_ap_df[["ux", "uy", "uz", "rotx", "roty", "rotz"]].to_numpy(dtype=np.float64)
    ap_vals = ap_vals.reshape(ncases, nap, 6)

    fbg_strains = batch_fbg_df[["eps_x", "eps_y", "eps_xy"]].to_numpy(dtype=np.float64)
    fbg_strains = fbg_strains.reshape(ncases, n_fbg, 3)

    array_disp = np.empty((ncases, n_elem, 3), dtype=np.float64)
    array_rot = np.empty((ncases, n_elem, 3), dtype=np.float64)

    for i in range(ncases):
        lin = LinearNDInterpolator(tri, ap_vals[i])
        interp = lin(target_xy)
        nan_mask = np.isnan(interp).any(axis=1)
        if np.any(nan_mask):
            interp[nan_mask] = ap_vals[i, near_idx[nan_mask]]
        array_disp[i] = interp[:, :3]
        array_rot[i] = interp[:, 3:]

    return fbg_strains, array_disp, array_rot, res


def load_upper_skin_nodes(csv_path: Path) -> pd.DataFrame:
    """Load upper-skin nodal coordinates, displacements, and rotations exported by MAPDL."""
    cols = ["node_id", "x", "y", "z", "ux", "uy", "uz", "rotx", "roty", "rotz"]
    df = pd.read_csv(csv_path, header=None, names=cols)
    df["node_id"] = df["node_id"].astype(int)
    return df


def load_fbg_sensors(csv_path: Path) -> pd.DataFrame:
    """Load 12 FBG surface strain readings (with microstrain helpers) exported by MAPDL."""
    cols = [
        "sensor_id",
        "x_target",
        "y_target",
        "x_node",
        "y_node",
        "eps_x",
        "eps_y",
        "eps_xy",
    ]
    df = pd.read_csv(csv_path, header=None, names=cols)
    df["sensor_id"] = df["sensor_id"].astype(int)
    df["ueps_x"] = df["eps_x"] * 1.0e6
    df["ueps_y"] = df["eps_y"] * 1.0e6
    df["ueps_xy"] = df["eps_xy"] * 1.0e6
    return df


def interpolate_to_array(
    upper_skin_csv: Path,
    array_cfg: ArrayConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate MAPDL upper-skin displacements & rotations onto all 256 AESA elements.

    Returns (disp_xyz, rot_xyz) each of shape (num_elements, 3) in SI units (m, rad).
    """
    df = load_upper_skin_nodes(upper_skin_csv)
    pts_xy = df[["x", "y"]].to_numpy(dtype=np.float64)
    vals = df[["ux", "uy", "uz", "rotx", "roty", "rotz"]].to_numpy(dtype=np.float64)

    target_xy = array_cfg.nominal_element_positions()[:, :2]
    lin = LinearNDInterpolator(pts_xy, vals)
    interp = lin(target_xy)

    nan_mask = np.isnan(interp).any(axis=1)
    if np.any(nan_mask):
        near = NearestNDInterpolator(pts_xy, vals)
        interp[nan_mask] = near(target_xy[nan_mask])

    return interp[:, :3], interp[:, 3:]

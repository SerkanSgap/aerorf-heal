"""Export a clean, zero-bloat GitHub upload bundle for AeroRF-Heal."""

from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parent / "aerorf-heal-github-upload"

EXCLUDE_DIRS = {
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".git",
    "doe_batches",
    "vv_cantilever",
    "wingbox_5g_thermal",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".rst",
    ".esav",
    ".full",
    ".dsp",
    ".lock",
    ".out",
    ".err",
    ".page",
    ".stat",
    ".mntr",
    ".log",
    ".bat",
    ".db",
    ".dbb",
}


def should_copy(src_file: Path, rel_parts: tuple[str, ...]) -> bool:
    """Return True if file belongs in the clean GitHub bundle."""
    if any(part in EXCLUDE_DIRS for part in rel_parts):
        return False
    if src_file.suffix.lower() in EXCLUDE_EXTS:
        return False
    if src_file.name in {"Thumbs.db", ".DS_Store", "tmp_epto.txt"}:
        return False
    return True


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True, exist_ok=True)

    copied = 0
    total_bytes = 0
    largest_file = ("", 0)

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if not should_copy(path, rel.parts):
            continue
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sz = path.stat().st_size
        copied += 1
        total_bytes += sz
        if sz > largest_file[1]:
            largest_file = (str(rel), sz)

    total_mb = total_bytes / (1024.0 * 1024.0)
    max_mb = largest_file[1] / (1024.0 * 1024.0)
    print(f"Exported GitHub bundle to: {DEST}")
    print(f"Files copied : {copied}")
    print(f"Total size   : {total_mb:.2f} MB")
    print(f"Largest file : {largest_file[0]} ({max_mb:.2f} MB)")

    if copied >= 100:
        raise RuntimeError(f"Expected < 100 files for easy GitHub web upload, got {copied}")
    if max_mb >= 25.0:
        raise RuntimeError(f"Largest file {largest_file[0]} exceeds GitHub Web 25 MB limit ({max_mb:.2f} MB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Import route annotations and Alpamayo prediction JSON files into pipeline/annotations.db.

Usage:
  python3 pipeline/import_route_db.py datasets/route_1
  python3 pipeline/import_route_db.py datasets/route_1/segment_00
  cd pipeline && python3 import_route_db.py ../datasets/route_1/segment_00
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run both pipeline DB import scripts for a route or segment."
    )
    parser.add_argument("target", help="Route folder or segment folder")
    parser.add_argument("--db", default=None, help="SQLite DB path. Default: pipeline/annotations.db")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Alpamayo predictions for matching frames.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
        candidates.append(PIPELINE_DIR / path)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return path.resolve()


def resolve_db_path(raw_path: str | None) -> Path:
    if raw_path is None:
        return PIPELINE_DIR / "annotations.db"
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "pipeline":
        return PROJECT_ROOT / path
    return path.resolve()


def discover_segments(target: Path) -> tuple[Path, list[Path]]:
    if not target.is_dir():
        raise SystemExit(f"[ERROR] Folder does not exist: {target}")

    if target.name.startswith("segment_"):
        return target.parent, [target]

    segments = sorted(path for path in target.iterdir() if path.is_dir() and path.name.startswith("segment_"))
    if not segments:
        raise SystemExit(f"[ERROR] No segment_* folders found under {target}")
    return target, segments


def run_command(command: list[str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    db_path = resolve_db_path(args.db)
    route_dir, segments = discover_segments(resolve_input_path(args.target))

    print(f"[INFO] Route folder: {route_dir}")
    print("[INFO] Segments:")
    for segment in segments:
        print(f"  - {segment}")

    for segment in segments:
        run_command(
            [
                sys.executable,
                str(PIPELINE_DIR / "import_route_annotations.py"),
                str(segment),
                "--db",
                str(db_path),
                *(["--dry-run"] if args.dry_run else []),
            ]
        )

    for segment in segments:
        run_command(
            [
                sys.executable,
                str(PIPELINE_DIR / "import_alpamayo_prediction_json.py"),
                str(segment),
                "--db",
                str(db_path),
                *(["--overwrite"] if args.overwrite else []),
                *(["--dry-run"] if args.dry_run else []),
            ]
        )

    print(f"[SUCCESS] Imported into {db_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run Alpamayo batch inference from the pipeline folder or project root.

Examples:
  python3 pipeline/run_alpamayo.py datasets/route_1
  python3 pipeline/run_alpamayo.py datasets/route_1/segment_00
  cd pipeline && python3 run_alpamayo.py ../datasets/route_1/segment_00
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
ALPAMAYO_BATCH_EXPORT = PROJECT_ROOT / "alpamayo" / "batch_export_inference.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Alpamayo batch_export_inference.py for a route or segment. Writes prediction JSON only."
    )
    parser.add_argument("target", help="Route folder or segment folder")
    parser.add_argument(
        "--segment",
        default=None,
        help="Optional segment name when target is a route folder, e.g. segment_00.",
    )
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        default=1,
        help="Number of trajectory samples. Default: 1.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Extra args passed through after --extra, e.g. --extra --start-frame 0 --end-frame 10.",
    )
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


def resolve_route_and_segment(target: Path, segment: str | None) -> tuple[Path, str | None]:
    if not target.exists():
        raise SystemExit(f"[ERROR] Folder does not exist: {target}")
    if not target.is_dir():
        raise SystemExit(f"[ERROR] Target must be a folder: {target}")

    if target.name.startswith("segment_"):
        if segment is not None and segment != target.name:
            raise SystemExit(
                f"[ERROR] Target is already {target.name}; do not also pass --segment {segment}."
            )
        return target.parent, target.name

    return target, segment


def main() -> None:
    args = parse_args()
    route_dir, segment_name = resolve_route_and_segment(resolve_input_path(args.target), args.segment)

    command = [
        sys.executable,
        str(ALPAMAYO_BATCH_EXPORT),
        "--route",
        str(route_dir),
        "--num-traj-samples",
        str(args.num_traj_samples),
    ]
    if segment_name:
        command.extend(["--segment", segment_name])
    if args.extra:
        command.extend(args.extra)

    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()

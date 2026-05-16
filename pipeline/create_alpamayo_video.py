#!/usr/bin/env python3
"""
Create an Alpamayo inference video from saved route dataset artifacts.

The renderer reads only:
  - segment raw frames
  - segment predictions/*_prediction.json
  - prediction JSON ground-truth and selected prediction paths
  - optional segment annotations
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
DATASET_FPS = 10.0
OUTPUT_FPS = 40.0

ANNOTATION_COLORS = [
    (57, 214, 248),
    (71, 166, 255),
    (102, 220, 105),
    (255, 166, 77),
    (214, 102, 255),
    (93, 234, 197),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an Alpamayo video from route raw frames, annotations, and prediction JSON."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Route folder or segment folder, e.g. datasets/route_1 or datasets/route_1/segment_00.",
    )
    parser.add_argument("--route", default=None, help="Route folder override.")
    parser.add_argument(
        "--segment",
        default=None,
        help="Optional segment name when target/--route is a route folder, e.g. segment_00.",
    )
    parser.add_argument("--start-frame", type=int, default=None, help="First frame index to render.")
    parser.add_argument("--end-frame", type=int, default=None, help="Last frame index to render.")
    parser.add_argument("--fps", type=float, default=OUTPUT_FPS, help="Output video FPS.")
    parser.add_argument(
        "--dataset-fps",
        type=float,
        default=DATASET_FPS,
        help="Dataset frame rate used to repeat frames for playback speed.",
    )
    parser.add_argument("--output", default=None, help="Output filename or path. Default is in the route folder.")
    parser.add_argument("--raw-dir", default="raw", help="Raw camera directory name. Default: raw.")
    parser.add_argument(
        "--prediction-frames",
        type=int,
        default=None,
        help=(
            "Maximum number of stored future trajectory points to draw. "
            "At the default dataset rate, 30 points is 3.0 seconds."
        ),
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


def resolve_route_and_segments(args: argparse.Namespace) -> tuple[Path, list[Path]]:
    raw_target = args.route or args.target
    if raw_target is None:
        raise SystemExit("[ERROR] Pass a route folder or segment folder.")

    target = resolve_input_path(raw_target)
    if not target.exists() or not target.is_dir():
        raise SystemExit(f"[ERROR] Folder does not exist: {target}")

    if target.name.startswith("segment_"):
        if args.segment is not None and args.segment != target.name:
            raise SystemExit(
                f"[ERROR] Target is already {target.name}; do not also pass --segment {args.segment}."
            )
        return target.parent, [target]

    route = target
    if args.segment:
        segment_dir = route / args.segment
        if not segment_dir.is_dir():
            raise SystemExit(f"[ERROR] Segment folder does not exist: {segment_dir}")
        return route, [segment_dir]

    segments = sorted(path for path in route.glob("segment_*") if path.is_dir())
    if not segments:
        raise SystemExit(f"[ERROR] No segment_* folders found in {route}")
    return route, segments


def frame_index_from_prediction(path: Path, payload: dict) -> int | None:
    frame_index = payload.get("frame_index")
    if frame_index is not None:
        return int(frame_index)

    stem = path.stem
    parts = stem.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return None


def load_predictions(segment_dir: Path) -> dict[int, dict]:
    predictions_dir = segment_dir / "predictions"
    predictions: dict[int, dict] = {}
    for json_path in sorted(predictions_dir.glob("*_prediction.json")):
        with json_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        frame_index = frame_index_from_prediction(json_path, payload)
        if frame_index is None:
            print(f"[SKIP] {json_path}: missing frame index")
            continue
        payload["_json_path"] = str(json_path)
        predictions[frame_index] = payload
    return predictions


def image_path_for_frame(raw_dir: Path, frame_index: int) -> Path | None:
    for ext in IMAGE_EXTS:
        path = raw_dir / f"{frame_index:06d}{ext}"
        if path.exists():
            return path
    return None


def read_classes(annotation_dir: Path) -> list[str]:
    candidates = [
        annotation_dir / "classes.txt",
        annotation_dir / "raw" / "classes.txt",
    ]
    for path in candidates:
        if path.exists():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []


def label_path_for_frame(segment_dir: Path, raw_dir_name: str, frame_index: int) -> Path | None:
    annotation_dir = segment_dir / "annotations"
    candidates = [
        annotation_dir / raw_dir_name / "labels" / f"{frame_index:06d}.txt",
        annotation_dir / "labels" / f"{frame_index:06d}.txt",
        segment_dir / "local_yolo_annotations" / raw_dir_name / "labels" / f"{frame_index:06d}.txt",
        segment_dir / "local_yolo_annotations" / "labels" / f"{frame_index:06d}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def draw_annotations(frame: np.ndarray, label_path: Path | None, classes: list[str]) -> None:
    if label_path is None:
        return

    height, width = frame.shape[:2]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:])
        x1 = int((xc - bw / 2.0) * width)
        y1 = int((yc - bh / 2.0) * height)
        x2 = int((xc + bw / 2.0) * width)
        y2 = int((yc + bh / 2.0) * height)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        color = ANNOTATION_COLORS[class_id % len(ANNOTATION_COLORS)]
        label = classes[class_id] if class_id < len(classes) else str(class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        text_w, text_h = text_size
        cv2.rectangle(frame, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 8, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(text_h + 2, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )


def path_points(records: list[dict], max_points: int | None = None) -> list[tuple[float, float]]:
    points = []
    for record in records[:max_points]:
        x_m = float(record.get("x_m", 0.0))
        y_m = float(record.get("y_m", 0.0))
        points.append((-y_m, x_m))
    return points


def draw_polyline(panel: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    if len(points) < 2:
        return
    cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(panel, point, 4, color, -1, cv2.LINE_AA)


def draw_path_panel(frame: np.ndarray, payload: dict, prediction_frames: int | None = None) -> None:
    height, width = frame.shape[:2]
    panel_w = min(380, max(280, width // 3))
    panel_h = min(340, max(260, height // 3))
    margin = 24
    x0 = max(margin, width - panel_w - margin)
    y0 = max(margin, height - panel_h - margin)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (245, 245, 245), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)

    panel = frame[y0 : y0 + panel_h, x0 : x0 + panel_w]
    pad = 34
    plot_w = panel_w - pad * 2
    plot_h = panel_h - pad * 2
    origin = (panel_w // 2, panel_h - pad)
    cv2.line(panel, (pad, origin[1]), (panel_w - pad, origin[1]), (190, 190, 190), 1)
    cv2.line(panel, (origin[0], pad), (origin[0], panel_h - pad), (190, 190, 190), 1)

    frames_stored = int(payload.get("frames_stored", len(payload.get("selected_path", []))))
    max_points = max(1, frames_stored)
    if prediction_frames is not None:
        max_points = min(max_points, max(1, prediction_frames))
    pred = path_points(payload.get("selected_path", []), max_points)
    gt = path_points(payload.get("ground_truth_path", []), max_points)
    all_points = pred + gt + [(0.0, 0.0)]
    max_abs_x = max(abs(point[0]) for point in all_points) if all_points else 5.0
    max_y = max(abs(point[1]) for point in all_points) if all_points else 5.0
    scale = min(plot_w / max(max_abs_x * 2.2, 10.0), plot_h / max(max_y * 1.15, 10.0))

    def to_px(point: tuple[float, float]) -> tuple[int, int]:
        lateral, forward = point
        return int(origin[0] + lateral * scale), int(origin[1] - forward * scale)

    draw_polyline(panel, [to_px(point) for point in gt], (45, 45, 235))
    draw_polyline(panel, [to_px(point) for point in pred], (45, 220, 45))
    cv2.drawMarker(panel, origin, (20, 20, 20), cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)

    cv2.putText(panel, "Ground Truth", (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 45, 235), 1, cv2.LINE_AA)
    cv2.putText(panel, "Prediction", (150, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 180, 45), 1, cv2.LINE_AA)


def draw_text_panel(frame: np.ndarray, segment_name: str, frame_index: int, payload: dict) -> None:
    command = payload.get("command_text") or payload.get("command") or payload.get("nav_command", "")
    reasoning = payload.get("reasoning_text") or payload.get("reasoning", "")

    lines = [
        f"{segment_name} | Frame {frame_index:06d}",
        f"Command: {command}",
    ]
    if reasoning:
        lines.extend(textwrap.wrap(f"Reasoning: {reasoning}", width=76))

    x = 24
    y = 34
    line_h = 24
    panel_w = min(frame.shape[1] - 48, 900)
    panel_h = min(frame.shape[0] - 48, 18 + line_h * min(len(lines), 10))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - 24), (x - 10 + panel_w, y - 24 + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    for idx, line in enumerate(lines[:10]):
        color = (255, 255, 255)
        if idx == 1:
            color = (110, 255, 110)
        elif idx >= 2:
            color = (80, 210, 255)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)
        y += line_h


def default_output_path(route_dir: Path, segments: list[Path], args: argparse.Namespace) -> Path:
    if args.output:
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = route_dir / output
        return output.resolve()

    route_name = route_dir.name
    if len(segments) == 1:
        base = f"{segments[0].name}_{route_name}_inference"
    else:
        base = f"{route_name}_inference"
    if args.start_frame is not None or args.end_frame is not None:
        start = args.start_frame if args.start_frame is not None else 0
        end = args.end_frame if args.end_frame is not None else "end"
        base = f"{base}_{start}_{end}"
    return (route_dir / f"{base}.mp4").resolve()


def render_video(route_dir: Path, segments: list[Path], args: argparse.Namespace) -> Path:
    if args.prediction_frames is not None and args.prediction_frames < 1:
        raise SystemExit("[ERROR] --prediction-frames must be at least 1.")

    output_path = default_output_path(route_dir, segments, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_repeat = max(1, int(round(args.fps / args.dataset_fps)))
    writer = None
    writer_size = None
    written = 0

    for segment_dir in segments:
        predictions = load_predictions(segment_dir)
        if not predictions:
            print(f"[SKIP] {segment_dir.name}: no prediction JSON files")
            continue

        frame_indices = sorted(predictions)
        if args.start_frame is not None:
            frame_indices = [idx for idx in frame_indices if idx >= args.start_frame]
        if args.end_frame is not None:
            frame_indices = [idx for idx in frame_indices if idx <= args.end_frame]

        raw_dir = segment_dir / args.raw_dir
        if not raw_dir.is_dir():
            print(f"[SKIP] {segment_dir.name}: raw frame folder not found: {raw_dir}")
            continue

        classes = read_classes(segment_dir / "annotations")
        print(f"Rendering {segment_dir.name}: {len(frame_indices)} prediction frame(s)")

        for frame_index in frame_indices:
            image_path = image_path_for_frame(raw_dir, frame_index)
            if image_path is None:
                print(f"[SKIP] {segment_dir.name} frame {frame_index}: raw frame missing")
                continue

            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[SKIP] {image_path}: could not read image")
                continue

            payload = predictions[frame_index]
            label_path = label_path_for_frame(segment_dir, args.raw_dir, frame_index)
            draw_annotations(frame, label_path, classes)
            draw_path_panel(frame, payload, args.prediction_frames)
            draw_text_panel(frame, segment_dir.name, frame_index, payload)

            if writer is None:
                height, width = frame.shape[:2]
                writer_size = (width, height)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, writer_size)
                if not writer.isOpened():
                    raise SystemExit(f"[ERROR] Failed to open video writer for {output_path}")
            elif writer_size is not None and (frame.shape[1], frame.shape[0]) != writer_size:
                frame = cv2.resize(frame, writer_size, interpolation=cv2.INTER_AREA)

            for _ in range(frame_repeat):
                writer.write(frame)
                written += 1

    if writer is not None:
        writer.release()

    if written == 0:
        raise SystemExit("[ERROR] No frames were written. Check that prediction JSON and raw frames exist.")

    print(f"Saved video: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    route_dir, segments = resolve_route_and_segments(args)
    render_video(route_dir, segments, args)


if __name__ == "__main__":
    main()

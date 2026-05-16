#!/usr/bin/env python3
"""
Loads dashcam frames (and optionally annotations) into annotations.db.

Usage:
    # Register the 39 aspave frames (no annotations yet)
    python import_annotations.py

    # Import frames + annotations from a JSON file
    python import_annotations.py --annotations-json path/to/annotations.json

    # Custom paths
    python import_annotations.py --frames-dir /path/to/frames --db my.db

Expected JSON annotation format:
    [
      {
        "filename": "aspave_frame_0000.jpg",
        "scene_description": "Clear highway with vehicles",
        "steering_angle_deg": 2.5,
        "throttle": 0.6,
        "brake": 0.0,
        "annotation_source": "chatgpt",   // optional, default "manual"
        "annotated_at": "2026-01-15T10:30:00",  // optional
        "labels": {
          "pedestrian":   false,
          "vehicle":      true,
          "traffic_light": false,
          "lane_marking": true,
          "obstacle":     false
        }
      },
      ...
    ]
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from dataset_manager import DatasetManager


_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_FRAMES_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "frame_extractor", "extracted_frames", "aspave")
)
DEFAULT_DB = os.path.join(_HERE, "annotations.db")


def get_image_dimensions(image_path: str):
    """Return (width, height) using PIL, or (None, None) if unavailable."""
    if PIL_AVAILABLE:
        try:
            with Image.open(image_path) as img:
                return img.width, img.height
        except Exception as e:
            print(f"  Warning: could not read dimensions for {image_path}: {e}")
    return None, None


def extract_frame_number(filename: str) -> int:
    """
    Extract numeric index from aspave_frame_XXXX.jpg filenames.
    Returns -1 if the pattern is not matched.
    """
    stem = Path(filename).stem          # e.g. "aspave_frame_0007"
    parts = stem.rsplit('_', 1)         # ['aspave_frame', '0007']
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return -1


def make_relative_path(filename: str) -> str:
    """
    Build the relative path from llm-model-tests/ to the frame file.
    Example: '../frame_extractor/extracted_frames/aspave/aspave_frame_0001.jpg'
    """
    return os.path.join(
        "..", "frame_extractor", "extracted_frames", "aspave", filename
    ).replace("\\", "/")


# Frame import

def import_frames(db: DatasetManager, frames_dir: str) -> dict:
    """
    Scan frames_dir for JPG files and register them in the database.

    Returns:
        {filename: frame_id} mapping for all registered frames.
    """
    frames_dir = os.path.abspath(frames_dir)
    if not os.path.isdir(frames_dir):
        print(f"ERROR: Frames directory not found: {frames_dir}")
        sys.exit(1)

    jpg_files = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg'))
    )

    if not jpg_files:
        print(f"No JPG files found in: {frames_dir}")
        return {}

    print(f"Found {len(jpg_files)} JPG file(s) in: {frames_dir}")
    frame_map: dict = {}

    for filename in jpg_files:
        abs_path = os.path.join(frames_dir, filename)
        relative_path = make_relative_path(filename)
        width, height = get_image_dimensions(abs_path)
        frame_number = extract_frame_number(filename)

        frame_id = db.add_frame(
            filename=filename,
            relative_path=relative_path,
            width=width,
            height=height,
            source='aspave',
            frame_number=frame_number,
        )
        frame_map[filename] = frame_id

        dim_str = f"{width}x{height}" if width else "dims unknown"
        print(f"  [id={frame_id:4d}] {filename}  ({dim_str})")

    return frame_map


# Annotation import

def import_annotations_from_json(
    db: DatasetManager, json_path: str, frame_map: dict
) -> int:
    """
    Load driving annotations from a JSON file and insert them.

    Returns:
        Number of annotations successfully inserted.
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("ERROR: JSON file must contain a top-level array.")
        return 0

    print(f"Loaded {len(data)} annotation entries from: {json_path}")
    inserted = 0
    skipped = 0

    for entry in data:
        filename = entry.get('filename')
        if not filename:
            print("  WARNING: Entry missing 'filename' — skipping.")
            skipped += 1
            continue

        frame_id = frame_map.get(filename)
        if not frame_id:
            print(f"  WARNING: No registered frame for '{filename}' — skipping.")
            skipped += 1
            continue

        try:
            ann_id = db.add_annotation(
                frame_id=frame_id,
                scene_description=entry.get('scene_description', ''),
                steering_angle_deg=float(entry['steering_angle_deg']),
                throttle=float(entry['throttle']),
                brake=float(entry['brake']),
                annotation_source=entry.get('annotation_source', 'manual'),
                annotated_at=entry.get('annotated_at'),
            )
        except (ValueError, KeyError) as e:
            print(f"  ERROR: Could not import annotation for '{filename}': {e}")
            skipped += 1
            continue

        # Import label categories if provided
        labels = entry.get('labels', {})
        for category, value in labels.items():
            # 'value' may be bool or dict {"present": bool, "confidence": float}
            if isinstance(value, dict):
                present = bool(value.get('present', False))
                confidence = value.get('confidence')
            else:
                present = bool(value)
                confidence = None
            try:
                db.add_label_category(ann_id, category, present, confidence)
            except ValueError as e:
                print(f"  WARNING: Label skipped — {e}")

        steering = entry['steering_angle_deg']
        throttle = entry['throttle']
        brake = entry['brake']
        print(
            f"  [ann={ann_id:4d}] {filename}  "
            f"steering={steering:+.1f}° throttle={throttle:.2f} brake={brake:.2f}"
        )
        inserted += 1

    print(f"\nInserted: {inserted}  |  Skipped: {skipped}")
    return inserted



def main():
    parser = argparse.ArgumentParser(
        description="Import dashcam frames (and optional annotations) into annotations.db"
    )
    parser.add_argument(
        "--frames-dir", default=DEFAULT_FRAMES_DIR,
        help=f"Directory of JPG frames (default: {DEFAULT_FRAMES_DIR})"
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"Path to SQLite database (default: {DEFAULT_DB})"
    )
    parser.add_argument(
        "--annotations-json", default=None,
        help="Optional JSON file with driving annotations to import"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be imported without writing to the database"
    )
    args = parser.parse_args()

    if not PIL_AVAILABLE:
        print("Warning: Pillow not installed — image dimensions will not be recorded.")
        print("         Install with: pip install Pillow\n")

    print("=" * 60)
    print("Annotations Database Import")
    print("=" * 60)
    print(f"Database   : {args.db}")
    print(f"Frames dir : {args.frames_dir}")
    if args.annotations_json:
        print(f"Annotations: {args.annotations_json}")
    if args.dry_run:
        print("DRY RUN -- no changes will be written.")
    print()

    if args.dry_run:
        print("[DRY RUN] Would scan:", args.frames_dir)
        jpgs = [
            f for f in os.listdir(args.frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg'))
        ] if os.path.isdir(args.frames_dir) else []
        print(f"[DRY RUN] Found {len(jpgs)} JPG files.")
        return

    with DatasetManager(args.db) as db:
        print("--- Registering frames ---")
        frame_map = import_frames(db, args.frames_dir)
        print(f"\nTotal frames registered: {len(frame_map)}")

        if args.annotations_json:
            if not os.path.exists(args.annotations_json):
                print(f"ERROR: Annotations file not found: {args.annotations_json}")
                sys.exit(1)
            print(f"\n--- Importing annotations ---")
            count = import_annotations_from_json(db, args.annotations_json, frame_map)
            print(f"Total annotations imported: {count}")
        else:
            print(
                "\nNo --annotations-json provided.\n"
                "Frames registered without driving labels.\n"
                "Run again with --annotations-json <file> to add labels."
            )

        print("\n--- Database summary ---")
        stats = db.get_stats()
        print(f"  Frames      : {stats['total_frames']}")
        print(f"  Annotations : {stats['total_annotations']}")
        if stats['by_source']:
            print(f"  By source   : {stats['by_source']}")
        if stats['label_counts']:
            print("  Label counts (present/total):")
            for cat, counts in stats['label_counts'].items():
                print(f"    {cat:<15}: {counts['present']}/{counts['total']}")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Import Alpamayo per-frame prediction JSON files into annotations.db.

This pairs with:
  alpamayo/batch_export_inference.py

Example:
  python3 llm-model-tests/import_alpamayo_prediction_json.py datasets/route_3/segment_00
"""

from __future__ import annotations

import argparse
import glob
import json
import sqlite3
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Import Alpamayo prediction JSON into SQLite.")
    parser.add_argument("segment", help="Segment folder, e.g. datasets/route_3/segment_00")
    parser.add_argument("--db", default="pipeline/annotations.db")
    parser.add_argument("--predictions-dir", default=None, help="Optional predictions folder override")
    parser.add_argument(
        "--source",
        default=None,
        help="Optional frame source override. Default is inferred as <route>_<segment>.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def connect_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_tables(conn)
    return conn


def ensure_tables(conn):
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS alpamayo_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
            annotation_id INTEGER REFERENCES annotations(id) ON DELETE SET NULL,
            model_name TEXT NOT NULL,
            nav_command TEXT NOT NULL,
            command_text TEXT,
            nav_command_source TEXT NOT NULL,
            selection_mode TEXT NOT NULL,
            selected_sample_index INTEGER NOT NULL,
            num_traj_samples INTEGER NOT NULL,
            guidance_weight REAL NOT NULL,
            max_generation_length INTEGER NOT NULL,
            frames_requested INTEGER NOT NULL,
            frames_stored INTEGER NOT NULL,
            reasoning_text TEXT,
            cot TEXT,
            selected_path_json TEXT NOT NULL,
            all_samples_json TEXT,
            gt_path_json TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS alpamayo_prediction_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL REFERENCES alpamayo_predictions(id) ON DELETE CASCADE,
            step_index INTEGER NOT NULL,
            x_m REAL NOT NULL,
            y_m REAL NOT NULL,
            z_m REAL NOT NULL,
            UNIQUE(prediction_id, step_index)
        );

        CREATE INDEX IF NOT EXISTS idx_alpamayo_predictions_frame_id
            ON alpamayo_predictions(frame_id);
        CREATE INDEX IF NOT EXISTS idx_alpamayo_prediction_points_prediction_id
            ON alpamayo_prediction_points(prediction_id);
        """
    )
    ensure_column(conn, "alpamayo_predictions", "command_text", "TEXT")
    ensure_column(conn, "alpamayo_predictions", "reasoning_text", "TEXT")
    conn.commit()


def ensure_column(conn, table, column, column_type):
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def load_prediction(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def infer_source(payload):
    route = payload.get("route")
    segment = payload.get("segment")
    if route and segment:
        return f"{route}_{segment}"
    return None


def infer_source_from_segment(segment_path):
    segment_path = Path(segment_path).resolve()
    return f"{segment_path.parent.name}_{segment_path.name}"


def infer_predictions_dir(segment_path):
    return Path(segment_path).resolve() / "predictions"


def find_frame(conn, source, frame_index):
    row = conn.execute(
        """
        SELECT f.id AS frame_id, MAX(a.id) AS annotation_id
        FROM frames f
        LEFT JOIN annotations a ON a.frame_id = f.id
        WHERE f.source = ? AND f.frame_number = ?
        GROUP BY f.id
        ORDER BY f.id
        LIMIT 1
        """,
        (source, frame_index),
    ).fetchone()
    return row


def delete_existing_predictions(conn, frame_id):
    conn.execute("DELETE FROM alpamayo_predictions WHERE frame_id = ?", (frame_id,))
    conn.commit()


def insert_prediction(conn, frame_row, payload):
    selected_path = payload.get("selected_path", [])
    all_samples = payload.get("all_samples", [])
    gt_path = payload.get("ground_truth_path", [])
    command_text = payload.get("command_text") or payload.get("command") or payload.get("nav_command", "")
    reasoning_text = payload.get("reasoning_text") or payload.get("reasoning", "")

    cur = conn.execute(
        """
        INSERT INTO alpamayo_predictions (
            frame_id, annotation_id, model_name, nav_command, command_text, nav_command_source,
            selection_mode, selected_sample_index, num_traj_samples, guidance_weight,
            max_generation_length, frames_requested, frames_stored, reasoning_text, cot,
            selected_path_json, all_samples_json, gt_path_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            frame_row["frame_id"],
            frame_row["annotation_id"],
            payload.get("model_name", "nvidia/Alpamayo-1.5-10B"),
            payload.get("nav_command", ""),
            command_text,
            payload.get("command_source", "unknown"),
            payload.get("selection_mode", "heuristic"),
            int(payload.get("selected_sample_index", 0)),
            int(payload.get("num_traj_samples", 0)),
            float(payload.get("guidance_weight", 0.0)),
            int(payload.get("max_generation_length", 0)),
            int(payload.get("frames_requested", len(selected_path))),
            int(payload.get("frames_stored", len(selected_path))),
            reasoning_text,
            reasoning_text,
            json.dumps(selected_path),
            json.dumps(all_samples),
            json.dumps(gt_path),
        ),
    )
    prediction_id = cur.lastrowid

    conn.executemany(
        """
        INSERT INTO alpamayo_prediction_points
            (prediction_id, step_index, x_m, y_m, z_m)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                prediction_id,
                int(point["step_index"]),
                float(point["x_m"]),
                float(point["y_m"]),
                float(point.get("z_m", 0.0)),
            )
            for point in selected_path
        ],
    )
    conn.commit()
    return prediction_id


def main():
    args = parse_args()
    predictions_dir = Path(args.predictions_dir).resolve() if args.predictions_dir else infer_predictions_dir(args.segment)
    default_source = args.source or infer_source_from_segment(args.segment)

    prediction_paths = sorted(glob.glob(str(predictions_dir / "*_prediction.json")))
    if not prediction_paths:
        print(f"[SKIP] No *_prediction.json files found in {predictions_dir}")
        return

    conn = connect_db(args.db)
    imported = 0
    skipped = 0

    print(f"Found {len(prediction_paths)} prediction JSON file(s).")
    print(f"Predictions: {predictions_dir}")
    print(f"Source     : {default_source}")
    for json_path in prediction_paths:
        payload = load_prediction(json_path)
        source = args.source or infer_source(payload) or default_source
        frame_index = payload.get("frame_index")

        if source is None or frame_index is None:
            print(f"[SKIP] {json_path}: missing source/frame_index")
            skipped += 1
            continue

        frame_row = find_frame(conn, source, int(frame_index))
        if frame_row is None:
            print(f"[SKIP] {json_path}: no DB frame for source={source}, frame={frame_index}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"[DRY] {json_path} -> frame_id={frame_row['frame_id']}")
            imported += 1
            continue

        if args.overwrite:
            delete_existing_predictions(conn, frame_row["frame_id"])

        prediction_id = insert_prediction(conn, frame_row, payload)
        print(f"[OK] {json_path} -> prediction_id={prediction_id}")
        imported += 1

    print(f"\nDone. Imported: {imported} | Skipped: {skipped}")
    conn.close()


if __name__ == "__main__":
    main()

import csv
import json
import mimetypes
import os
import random
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Validation (mirrors Driving Instructions Angle Test.py)

MIN_TURN_ANGLE = -180
MAX_TURN_ANGLE = 180

VALID_CATEGORIES = ('pedestrian', 'vehicle', 'traffic_light', 'lane_marking', 'obstacle')


def validate_turn_angle(angle) -> bool:
    """Return True if angle is a number in [-180, 180] (inclusive)."""
    return isinstance(angle, (float, int)) and MIN_TURN_ANGLE <= angle <= MAX_TURN_ANGLE


def validate_throttle(value) -> bool:
    return isinstance(value, (float, int)) and 0.0 <= value <= 1.0


def validate_brake(value) -> bool:
    return isinstance(value, (float, int)) and 0.0 <= value <= 1.0


# Inline schema (fallback if schema.sql is missing)

_INLINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT    NOT NULL UNIQUE,
    relative_path TEXT    NOT NULL,
    image_data    BLOB,
    image_mime_type TEXT,
    image_size_bytes INTEGER,
    width         INTEGER,
    height        INTEGER,
    source        TEXT    DEFAULT 'aspave',
    frame_number  INTEGER,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS annotations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id           INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    scene_description  TEXT,
    steering_angle_deg REAL    NOT NULL,
    throttle           REAL    NOT NULL,
    brake              REAL    NOT NULL,
    annotation_source  TEXT    NOT NULL DEFAULT 'manual',
    annotated_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    created_at         TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at         TEXT    NOT NULL DEFAULT (datetime('now')),
    CONSTRAINT chk_steering CHECK (steering_angle_deg >= -180.0 AND steering_angle_deg <= 180.0),
    CONSTRAINT chk_throttle CHECK (throttle >= 0.0 AND throttle <= 1.0),
    CONSTRAINT chk_brake    CHECK (brake    >= 0.0 AND brake    <= 1.0)
);

CREATE TABLE IF NOT EXISTS label_categories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    annotation_id INTEGER NOT NULL REFERENCES annotations(id) ON DELETE CASCADE,
    category      TEXT    NOT NULL,
    present       INTEGER NOT NULL DEFAULT 0,
    confidence    REAL,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    CONSTRAINT chk_category CHECK (
        category IN ('pedestrian','vehicle','traffic_light','lane_marking','obstacle')
    ),
    UNIQUE (annotation_id, category)
);

CREATE INDEX IF NOT EXISTS idx_annotations_frame_id ON annotations(frame_id);
CREATE INDEX IF NOT EXISTS idx_label_categories_annotation_id ON label_categories(annotation_id);
CREATE INDEX IF NOT EXISTS idx_label_categories_category ON label_categories(category);
CREATE INDEX IF NOT EXISTS idx_frames_frame_number ON frames(frame_number);
CREATE INDEX IF NOT EXISTS idx_frames_source ON frames(source);
"""


# DatasetManager

class DatasetManager:
    """
    CRUD interface for the annotations SQLite database.

    Usage:
        with DatasetManager("annotations.db") as db:
            frame_id = db.add_frame("aspave_frame_0001.jpg", "../frames/aspave_frame_0001.jpg")
            ann_id   = db.add_annotation(frame_id, "Clear road", 5.0, 0.6, 0.0)
            db.add_label_category(ann_id, "vehicle", present=True)
    """

    def __init__(self, db_path: str = "annotations.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()

    # Connection management

    def _connect(self) -> None:
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # Schema init

    def _create_tables(self) -> None:
        schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path) as f:
                sql = f.read()
        else:
            sql = _INLINE_SCHEMA
        self.conn.executescript(sql)
        self.conn.commit()

    # Frame operations

    def add_frame(
        self,
        filename: str,
        relative_path: str,
        width: int = None,
        height: int = None,
        source: str = 'aspave',
        frame_number: int = None,
        image_data: bytes = None,
        image_mime_type: str = None,
        image_size_bytes: int = None,
        image_path: str = None,
    ) -> int:
        """
        Insert a frame record. Returns the frame id.
        If filename already exists, returns its existing id (idempotent).
        Relative paths only — never pass absolute paths.
        """
        # Reject both OS-native absolute paths and Unix-style /absolute paths
        if os.path.isabs(relative_path) or relative_path.startswith('/'):
            raise ValueError(f"relative_path must not be absolute: {relative_path}")

        if image_data is None and image_path:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            image_size_bytes = len(image_data)
            image_mime_type = image_mime_type or mimetypes.guess_type(image_path)[0]

        if image_data is not None and image_size_bytes is None:
            image_size_bytes = len(image_data)

        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """INSERT OR IGNORE INTO frames
               (filename, relative_path, image_data, image_mime_type, image_size_bytes,
                width, height, source, frame_number, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                filename, relative_path, image_data, image_mime_type, image_size_bytes,
                width, height, source, frame_number, now, now,
            ),
        )
        self.conn.commit()

        row = self.conn.execute(
            "SELECT id FROM frames WHERE filename = ?", (filename,)
        ).fetchone()
        return row["id"]

    def get_frame(self, frame_id: int) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM frames WHERE id = ?", (frame_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_frame_by_filename(self, filename: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM frames WHERE filename = ?", (filename,)
        ).fetchone()
        return dict(row) if row else None

    def list_frames(self) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM frames ORDER BY frame_number"
        ).fetchall()
        return [dict(r) for r in rows]

    # Annotations

    def add_annotation(
        self,
        frame_id: int,
        scene_description: str,
        steering_angle_deg: float,
        throttle: float,
        brake: float,
        annotation_source: str = 'manual',
        annotated_at: str = None,
    ) -> int:
        """
        Insert a driving annotation. Returns the annotation id.
        Raises ValueError if steering/throttle/brake are out of range.
        """
        if not validate_turn_angle(steering_angle_deg):
            raise ValueError(
                f"Invalid steering_angle_deg: {steering_angle_deg} (must be -180 to 180)"
            )
        if not validate_throttle(throttle):
            raise ValueError(f"Invalid throttle: {throttle} (must be 0.0 to 1.0)")
        if not validate_brake(brake):
            raise ValueError(f"Invalid brake: {brake} (must be 0.0 to 1.0)")

        now = datetime.utcnow().isoformat()
        annotated_at = annotated_at or now

        cur = self.conn.execute(
            """INSERT INTO annotations
               (frame_id, scene_description, steering_angle_deg, throttle, brake,
                annotation_source, annotated_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                frame_id, scene_description, steering_angle_deg,
                throttle, brake, annotation_source, annotated_at, now, now,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_annotation(self, annotation_id: int) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM annotations WHERE id = ?", (annotation_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_annotations_for_frame(self, frame_id: int) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM annotations WHERE frame_id = ? ORDER BY id", (frame_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def update_annotation(self, annotation_id: int, **kwargs) -> None:
        """
        Update annotation fields. Validates steering/throttle/brake if provided.
        Allowed keys: scene_description, steering_angle_deg, throttle, brake, annotation_source
        """
        if 'steering_angle_deg' in kwargs and not validate_turn_angle(kwargs['steering_angle_deg']):
            raise ValueError(f"Invalid steering_angle_deg: {kwargs['steering_angle_deg']}")
        if 'throttle' in kwargs and not validate_throttle(kwargs['throttle']):
            raise ValueError(f"Invalid throttle: {kwargs['throttle']}")
        if 'brake' in kwargs and not validate_brake(kwargs['brake']):
            raise ValueError(f"Invalid brake: {kwargs['brake']}")

        allowed = {
            'scene_description', 'steering_angle_deg', 'throttle',
            'brake', 'annotation_source',
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return

        updates['updated_at'] = datetime.utcnow().isoformat()
        set_clause = ', '.join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [annotation_id]
        self.conn.execute(
            f"UPDATE annotations SET {set_clause} WHERE id = ?", values
        )
        self.conn.commit()

    def delete_frame(self, frame_id: int) -> None:
        """Delete a frame and all its annotations/labels (CASCADE)."""
        self.conn.execute("DELETE FROM frames WHERE id = ?", (frame_id,))
        self.conn.commit()

    def get_all_annotations(self) -> List[Dict]:
        """Return all annotations joined with frame info."""
        rows = self.conn.execute(
            """SELECT f.filename, f.relative_path, f.frame_number, f.source,
                      a.id AS annotation_id, a.scene_description,
                      a.steering_angle_deg, a.throttle, a.brake,
                      a.annotation_source, a.annotated_at
               FROM frames f
               JOIN annotations a ON a.frame_id = f.id
               ORDER BY f.frame_number"""
        ).fetchall()
        return [dict(r) for r in rows]

    # Label category operations

    def add_label_category(
        self,
        annotation_id: int,
        category: str,
        present: bool = True,
        confidence: float = None,
    ) -> int:
        """
        Add or update a label category for an annotation. Returns id.
        category must be one of: pedestrian, vehicle, traffic_light, lane_marking, obstacle
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category: '{category}'. Must be one of {VALID_CATEGORIES}"
            )

        now = datetime.utcnow().isoformat()
        cur = self.conn.execute(
            """INSERT OR REPLACE INTO label_categories
               (annotation_id, category, present, confidence, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (annotation_id, category, 1 if present else 0, confidence, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_labels_for_annotation(self, annotation_id: int) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM label_categories WHERE annotation_id = ? ORDER BY category",
            (annotation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # Query operations

    def get_frames_by_label(self, category: str) -> List[Dict]:
        """
        Return all frames that have the given label category marked as present (= 1).

        Example:
            frames = db.get_frames_by_label("pedestrian")
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category: '{category}'. Must be one of {VALID_CATEGORIES}"
            )

        rows = self.conn.execute(
            """SELECT DISTINCT f.*,
                      a.id AS annotation_id, a.scene_description,
                      a.steering_angle_deg, a.throttle, a.brake
               FROM frames f
               JOIN annotations a ON a.frame_id = f.id
               JOIN label_categories lc ON lc.annotation_id = a.id
               WHERE lc.category = ? AND lc.present = 1
               ORDER BY f.frame_number""",
            (category,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_by_description(self, keyword: str) -> List[Dict]:
        """
        Return frames whose scene_description contains keyword (case-insensitive).

        Example:
            results = db.search_by_description("pedestrian")
        """
        rows = self.conn.execute(
            """SELECT f.*, a.id AS annotation_id,
                      a.scene_description, a.steering_angle_deg, a.throttle, a.brake
               FROM frames f
               JOIN annotations a ON a.frame_id = f.id
               WHERE LOWER(a.scene_description) LIKE LOWER(?)
               ORDER BY f.frame_number""",
            (f"%{keyword}%",),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_train_val_split(
        self, val_ratio: float = 0.2, seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split all annotated frames into train / validation sets.

        Returns:
            (train_records, val_records) — no overlap, deterministic with seed.
        """
        rows = self.conn.execute(
            """SELECT DISTINCT f.id AS frame_id, f.filename, f.relative_path, f.frame_number,
                      a.id AS annotation_id, a.scene_description,
                      a.steering_angle_deg, a.throttle, a.brake, a.annotation_source
               FROM frames f
               JOIN annotations a ON a.frame_id = f.id
               ORDER BY f.frame_number"""
        ).fetchall()

        records = [dict(r) for r in rows]
        rng = random.Random(seed)
        shuffled = records.copy()
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - val_ratio))
        return shuffled[:split_idx], shuffled[split_idx:]

    def validate_all_steering_angles(self) -> List[Dict]:
        """
        Return any annotations whose steering_angle_deg is outside [-180, 180].
        An empty list means all angles are valid.
        """
        rows = self.conn.execute(
            """SELECT a.id, a.frame_id, a.steering_angle_deg, f.filename
               FROM annotations a
               JOIN frames f ON f.id = a.frame_id
               WHERE a.steering_angle_deg < -180 OR a.steering_angle_deg > 180"""
        ).fetchall()
        return [dict(r) for r in rows]

    # Export operations

    def _fetch_export_rows(self, frame_ids: Optional[List[int]] = None):
        if frame_ids:
            placeholders = ','.join('?' * len(frame_ids))
            return self.conn.execute(
                f"""SELECT f.filename, f.relative_path, f.frame_number, f.source,
                           f.width, f.height,
                           a.id AS annotation_id, a.scene_description,
                           a.steering_angle_deg, a.throttle, a.brake,
                           a.annotation_source, a.annotated_at
                    FROM frames f
                    JOIN annotations a ON a.frame_id = f.id
                    WHERE f.id IN ({placeholders})
                    ORDER BY f.frame_number""",
                frame_ids,
            ).fetchall()
        return self.conn.execute(
            """SELECT f.filename, f.relative_path, f.frame_number, f.source,
                      f.width, f.height,
                      a.id AS annotation_id, a.scene_description,
                      a.steering_angle_deg, a.throttle, a.brake,
                      a.annotation_source, a.annotated_at
               FROM frames f
               JOIN annotations a ON a.frame_id = f.id
               ORDER BY f.frame_number"""
        ).fetchall()

    def _attach_labels(self, record: Dict) -> Dict:
        labels = self.conn.execute(
            "SELECT category, present, confidence FROM label_categories WHERE annotation_id = ?",
            (record['annotation_id'],),
        ).fetchall()
        record['labels'] = {
            l['category']: {'present': bool(l['present']), 'confidence': l['confidence']}
            for l in labels
        }
        return record

    def export_to_json(
        self, output_path: str, frame_ids: Optional[List[int]] = None
    ) -> int:
        """
        Export annotations (with labels) to a JSON file.
        Returns number of records written.

        Example:
            count = db.export_to_json("dataset.json")
        """
        rows = self._fetch_export_rows(frame_ids)
        records = [self._attach_labels(dict(r)) for r in rows]

        with open(output_path, 'w') as f:
            json.dump(records, f, indent=2)

        return len(records)

    def export_to_csv(
        self, output_path: str, frame_ids: Optional[List[int]] = None
    ) -> int:
        """
        Export annotations to a CSV file (no label columns — use JSON for full data).
        Returns number of records written.
        """
        rows = self._fetch_export_rows(frame_ids)
        if not rows:
            return 0

        fieldnames = list(rows[0].keys())
        # Drop annotation_id from CSV (internal key)
        fieldnames = [f for f in fieldnames if f != 'annotation_id']

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows([dict(r) for r in rows])

        return len(rows)

    def export_train_val_split_json(
        self,
        output_dir: str,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[str, str]:
        """
        Export 80/20 train/val split to train.json and val.json.
        Returns (train_path, val_path).

        Example:
            train_path, val_path = db.export_train_val_split_json("splits/")
        """
        os.makedirs(output_dir, exist_ok=True)
        train_records, val_records = self.get_train_val_split(val_ratio, seed)

        def enrich(records: List[Dict]) -> List[Dict]:
            return [self._attach_labels(r) for r in records]

        train_path = os.path.join(output_dir, "train.json")
        val_path = os.path.join(output_dir, "val.json")

        with open(train_path, 'w') as f:
            json.dump(enrich(train_records), f, indent=2)
        with open(val_path, 'w') as f:
            json.dump(enrich(val_records), f, indent=2)

        print(f"Train: {len(train_records)} records -> {train_path}")
        print(f"Val:   {len(val_records)} records -> {val_path}")

        return train_path, val_path

    def get_stats(self) -> Dict:
        """Return summary statistics about the database contents."""
        frame_count = self.conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        ann_count = self.conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
        sources = self.conn.execute(
            "SELECT annotation_source, COUNT(*) AS cnt FROM annotations GROUP BY annotation_source"
        ).fetchall()
        label_counts = self.conn.execute(
            """SELECT category, SUM(present) AS present_count, COUNT(*) AS total
               FROM label_categories GROUP BY category"""
        ).fetchall()

        return {
            "total_frames": frame_count,
            "total_annotations": ann_count,
            "by_source": {r["annotation_source"]: r["cnt"] for r in sources},
            "label_counts": {
                r["category"]: {"present": r["present_count"], "total": r["total"]}
                for r in label_counts
            },
        }

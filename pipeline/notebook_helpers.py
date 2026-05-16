from __future__ import annotations

import io
import json
import sqlite3
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


COLORS = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#00acc1"]


def default_db_path() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "annotations.db").exists() and cwd.name == "pipeline":
        return cwd / "annotations.db"
    return cwd / "pipeline" / "annotations.db"


class DatabaseExplorer:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path).expanduser().resolve() if db_path else default_db_path()
        self.project_root = self.db_path.parent.parent
        self.db_dir = self.db_path.parent
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def table_exists(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def plot_counts(self):
        summary = []
        for table in [
            "frames",
            "annotations",
            "label_categories",
            "alpamayo_predictions",
            "alpamayo_prediction_points",
        ]:
            if self.table_exists(table):
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                summary.append({"table": table, "rows": count})

        if not summary:
            print("No known tables found.")
            return

        df_counts = pd.DataFrame(summary)
        ax = df_counts.plot.bar(x="table", y="rows", legend=False, figsize=(9, 4), color="#3b82f6")
        ax.set_xlabel("")
        ax.set_ylabel("rows")
        ax.set_title("Database contents")
        ax.tick_params(axis="x", rotation=35)
        plt.tight_layout()
        plt.show()

    def get_frame_row(self, frame_id_or_row):
        if isinstance(frame_id_or_row, int):
            row = self.conn.execute("SELECT * FROM frames WHERE id=?", (frame_id_or_row,)).fetchone()
        else:
            row = frame_id_or_row
        if row is None:
            raise ValueError("Frame not found")
        return row

    def resolve_frame_path(self, frame_id_or_row) -> Path:
        row = self.get_frame_row(frame_id_or_row)
        return (self.db_dir / row["relative_path"]).resolve()

    def load_frame_image(self, frame_id_or_row):
        row = self.get_frame_row(frame_id_or_row)
        if "image_data" in row.keys() and row["image_data"] is not None:
            return Image.open(io.BytesIO(row["image_data"])).convert("RGB")
        return Image.open(self.resolve_frame_path(row)).convert("RGB")

    def frame_labels(self, frame_id: int) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT lc.category, lc.present, lc.confidence
            FROM annotations a
            JOIN label_categories lc ON lc.annotation_id = a.id
            WHERE a.frame_id = ?
            ORDER BY lc.category
            """,
            (frame_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def annotation_paths(self, image_path: Path):
        camera_name = image_path.parent.name
        segment_dir = image_path.parent.parent if camera_name.startswith("raw") else image_path.parent
        ann_dir = segment_dir / "annotations"
        if not ann_dir.exists():
            ann_dir = segment_dir / "local_yolo_annotations"

        camera_preview = ann_dir / camera_name / "preview" / f"{image_path.stem}_annotated.jpg"
        flat_preview = ann_dir / "preview" / f"{image_path.stem}_annotated.jpg"
        camera_label_file = ann_dir / camera_name / "labels" / f"{image_path.stem}.txt"
        flat_label_file = ann_dir / "labels" / f"{image_path.stem}.txt"
        root_classes = ann_dir / "classes.txt"
        camera_classes = ann_dir / camera_name / "classes.txt"

        preview = camera_preview if camera_preview.exists() else flat_preview
        label_file = camera_label_file if camera_label_file.exists() else flat_label_file
        classes_file = root_classes if root_classes.exists() else camera_classes
        return preview, label_file, classes_file

    def draw_yolo_boxes(self, image_path: Path, label_file: Path, classes_file: Path):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        classes = []
        if classes_file.exists():
            classes = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]

        width, height = image.size
        if not label_file.exists():
            return image

        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height
            color = COLORS[class_id % len(COLORS)]
            label = classes[class_id] if class_id < len(classes) else str(class_id)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            text_box = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(
                (text_box[0] - 2, text_box[1] - 2, text_box[2] + 4, text_box[3] + 4),
                fill=color,
            )
            draw.text((x1, y1), label, fill="white", font=font)
        return image

    def show_frame(self, frame_id: int, figsize=(12, 7)):
        row = self.get_frame_row(frame_id)
        image_path = self.resolve_frame_path(row)
        preview, label_file, classes_file = self.annotation_paths(image_path)
        present = [item["category"] for item in self.frame_labels(frame_id) if item["present"]]

        if preview.exists():
            image = Image.open(preview).convert("RGB")
        elif label_file.exists() and classes_file.exists():
            image = self.draw_yolo_boxes(image_path, label_file, classes_file)
        else:
            image = self.load_frame_image(row)

        plt.figure(figsize=figsize)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"frame_id={frame_id} | {row['source']} #{row['frame_number']} | labels={present}")
        plt.show()

    def first_frame_id(self) -> int | None:
        row = self.conn.execute("SELECT id FROM frames ORDER BY source, frame_number LIMIT 1").fetchone()
        return int(row["id"]) if row else None

    def gallery_frame_ids(self, source: str | None = None, limit: int = 6) -> list[int]:
        if source:
            rows = self.conn.execute(
                "SELECT id FROM frames WHERE source = ? ORDER BY frame_number LIMIT ?",
                (source, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id FROM frames ORDER BY source, frame_number LIMIT ?",
                (limit,),
            ).fetchall()
        return [int(row["id"]) for row in rows]

    def show_first_frame(self):
        frame_id = self.first_frame_id()
        if frame_id is None:
            print("No frames found.")
            return
        self.show_frame(frame_id)

    def show_gallery(self, source: str | None = None, limit: int = 6):
        for frame_id in self.gallery_frame_ids(source=source, limit=limit):
            self.show_frame(frame_id, figsize=(10, 5))

    def prediction_points(self, prediction_id: int):
        return pd.read_sql_query(
            """
            SELECT step_index, x_m, y_m, z_m
            FROM alpamayo_prediction_points
            WHERE prediction_id = ?
            ORDER BY step_index
            """,
            self.conn,
            params=[prediction_id],
        )

    def load_prediction(self, prediction_id: int) -> dict:
        row = self.conn.execute(
            """
            SELECT ap.*, f.filename, f.source, f.frame_number
            FROM alpamayo_predictions ap
            JOIN frames f ON f.id = ap.frame_id
            WHERE ap.id = ?
            """,
            (prediction_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"No prediction_id={prediction_id}")
        return dict(row)

    def prediction_command_text(self, pred: dict) -> str:
        return (
            pred.get("command_text")
            or pred.get("command")
            or pred.get("nav_command")
            or ""
        )

    def prediction_reasoning_text(self, pred: dict) -> str:
        return pred.get("reasoning_text") or pred.get("reasoning") or pred.get("cot") or ""

    def latest_prediction_id(self) -> int | None:
        if not self.table_exists("alpamayo_predictions"):
            return None
        row = self.conn.execute("SELECT id FROM alpamayo_predictions ORDER BY id DESC LIMIT 1").fetchone()
        return int(row["id"]) if row else None

    def nav_prediction_ids(self, kind: str, limit: int = 3) -> list[int]:
        if not self.table_exists("alpamayo_predictions"):
            return []
        rows = self.conn.execute(
            """
            SELECT ap.id
            FROM alpamayo_predictions ap
            JOIN frames f ON f.id = ap.frame_id
            WHERE LOWER(ap.nav_command) LIKE ?
            ORDER BY f.source, f.frame_number, ap.id
            LIMIT ?
            """,
            (f"%{kind.lower()}%", limit),
        ).fetchall()
        return [int(row["id"]) for row in rows]

    def plot_prediction(self, prediction_id: int, show_gt: bool = True, show_image: bool = True):
        pred = self.load_prediction(prediction_id)
        pts = self.prediction_points(prediction_id)
        command_text = self.prediction_command_text(pred)
        reasoning_text = self.prediction_reasoning_text(pred)

        if show_image:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            image_path = self.resolve_frame_path(pred["frame_id"])
            preview, label_file, classes_file = self.annotation_paths(image_path)
            if preview.exists():
                image = Image.open(preview).convert("RGB")
            elif label_file.exists() and classes_file.exists():
                image = self.draw_yolo_boxes(image_path, label_file, classes_file)
            else:
                image = self.load_frame_image(pred["frame_id"])
            axes[0].imshow(image)
            axes[0].axis("off")
            ax = axes[1]
        else:
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(-pts["y_m"], pts["x_m"], marker="o", linewidth=2.0, color="lime", label="Selected Prediction")
        x_values = [-float(y) for y in pts["y_m"].tolist()]
        y_values = [float(x) for x in pts["x_m"].tolist()]

        if show_gt and pred.get("gt_path_json"):
            gt = pd.DataFrame(json.loads(pred["gt_path_json"]))
            if not gt.empty:
                ax.plot(-gt["y_m"], gt["x_m"], marker="o", linewidth=2.0, color="red", label="Ground Truth")
                x_values.extend([-float(y) for y in gt["y_m"].tolist()])
                y_values.extend([float(x) for x in gt["x_m"].tolist()])

        ax.plot(0, 0, marker="*", color="black", markersize=14)
        x_values.append(0.0)
        y_values.append(0.0)

        if x_values and y_values:
            min_x, max_x = min(x_values), max(x_values)
            min_y, max_y = min(y_values), max(y_values)
            x_mid = (min_x + max_x) / 2.0
            y_mid = (min_y + max_y) / 2.0
            x_half = max((max_x - min_x) / 2.0, 5.0)
            y_half = max((max_y - min_y) / 2.0, 5.0)
            ax.set_xlim(x_mid - x_half, x_mid + x_half)
            ax.set_ylim(y_mid - y_half, y_mid + y_half)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("-lateral y (m)")
        ax.set_ylabel("forward x (m)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.set_title(
            f"prediction_id={prediction_id} | frame_id={pred['frame_id']} | "
            f"{pred['source']} #{pred['frame_number']}\ncommand={command_text}"
        )
        if reasoning_text:
            wrapped_reasoning = "\n".join(
                textwrap.wrap(f"Reasoning: {reasoning_text}", width=125)
            )
            fig.subplots_adjust(bottom=0.18)
            fig.text(0.05, 0.03, wrapped_reasoning, ha="left", va="bottom", fontsize=9)
        plt.show()

    def plot_latest_prediction(self):
        prediction_id = self.latest_prediction_id()
        if prediction_id is None:
            print("No predictions found.")
            return
        self.plot_prediction(prediction_id)

    def plot_turn_predictions(self, limit: int = 3):
        left_ids = self.nav_prediction_ids("left", limit=limit)
        right_ids = self.nav_prediction_ids("right", limit=limit)
        if not left_ids and not right_ids:
            print("No left/right navigation predictions found yet.")
            return
        for prediction_id in left_ids + right_ids:
            self.plot_prediction(prediction_id)


def open_explorer(db_path: str | Path | None = None) -> DatabaseExplorer:
    explorer = DatabaseExplorer(db_path)
    print("Database:", explorer.db_path)
    return explorer

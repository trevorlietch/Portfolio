#!/usr/bin/env python3
"""
test_database.py - Ticket #14: Data Storage
Autonomous Vehicle Capstone Project

Unit tests for dataset_manager.py.

Covers the four required test scenarios:
  1. Insert 5 sample annotations
  2. Query all frames with 'pedestrian' labels
  3. Export 80/20 train/val split to JSON
  4. Verify all steering angles are valid

Run:
    python test_database.py
    python -m pytest test_database.py -v
"""

import json
import os
import tempfile
import unittest

from dataset_manager import (
    DatasetManager,
    VALID_CATEGORIES,
    validate_turn_angle,
    validate_throttle,
    validate_brake,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sample data (5 annotations)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_ANNOTATIONS = [
    {
        "filename": "aspave_frame_0000.jpg",
        "scene_description": "Clear highway with lane markings visible",
        "steering_angle_deg": 2.5,
        "throttle": 0.6,
        "brake": 0.0,
        "annotation_source": "chatgpt",
        "labels": {
            "pedestrian": False, "vehicle": True,
            "traffic_light": False, "lane_marking": True, "obstacle": False,
        },
    },
    {
        "filename": "aspave_frame_0001.jpg",
        "scene_description": "Pedestrian crossing ahead, slowing down",
        "steering_angle_deg": 0.0,
        "throttle": 0.2,
        "brake": 0.5,
        "annotation_source": "manual",
        "labels": {
            "pedestrian": True, "vehicle": False,
            "traffic_light": True, "lane_marking": True, "obstacle": False,
        },
    },
    {
        "filename": "aspave_frame_0002.jpg",
        "scene_description": "Right turn at intersection with vehicles waiting",
        "steering_angle_deg": 45.0,
        "throttle": 0.3,
        "brake": 0.1,
        "annotation_source": "chatgpt",
        "labels": {
            "pedestrian": False, "vehicle": True,
            "traffic_light": True, "lane_marking": True, "obstacle": False,
        },
    },
    {
        "filename": "aspave_frame_0003.jpg",
        "scene_description": "Obstacle on road, emergency stop",
        "steering_angle_deg": -10.0,
        "throttle": 0.0,
        "brake": 1.0,
        "annotation_source": "manual",
        "labels": {
            "pedestrian": False, "vehicle": False,
            "traffic_light": False, "lane_marking": True, "obstacle": True,
        },
    },
    {
        "filename": "aspave_frame_0004.jpg",
        "scene_description": "Pedestrian on sidewalk, normal cruise speed",
        "steering_angle_deg": -5.0,
        "throttle": 0.7,
        "brake": 0.0,
        "annotation_source": "chatgpt",
        "labels": {
            "pedestrian": True, "vehicle": True,
            "traffic_light": False, "lane_marking": True, "obstacle": False,
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Test: validate_turn_angle (mirrors Driving Instructions Angle Test.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateTurnAngle(unittest.TestCase):
    """validate_turn_angle() must match the logic in Driving Instructions Angle Test.py"""

    def test_valid_angle(self):
        self.assertTrue(validate_turn_angle(100))

    def test_invalid_angle(self):
        self.assertFalse(validate_turn_angle(200))

    def test_boundary_max(self):
        self.assertTrue(validate_turn_angle(180))

    def test_boundary_min(self):
        self.assertTrue(validate_turn_angle(-180))

    def test_edge_high(self):
        self.assertFalse(validate_turn_angle(181))

    def test_edge_low(self):
        self.assertFalse(validate_turn_angle(-181))

    def test_zero(self):
        self.assertTrue(validate_turn_angle(0))

    def test_float(self):
        self.assertTrue(validate_turn_angle(45.5))

    def test_string_rejected(self):
        self.assertFalse(validate_turn_angle("90"))

    def test_none_rejected(self):
        self.assertFalse(validate_turn_angle(None))

    def test_throttle_valid(self):
        self.assertTrue(validate_throttle(0.0))
        self.assertTrue(validate_throttle(1.0))
        self.assertTrue(validate_throttle(0.5))

    def test_throttle_invalid(self):
        self.assertFalse(validate_throttle(1.1))
        self.assertFalse(validate_throttle(-0.1))

    def test_brake_valid(self):
        self.assertTrue(validate_brake(0.0))
        self.assertTrue(validate_brake(1.0))

    def test_brake_invalid(self):
        self.assertFalse(validate_brake(2.0))


# ─────────────────────────────────────────────────────────────────────────────
# Base test class: in-memory DB with 5 sample annotations pre-loaded
# ─────────────────────────────────────────────────────────────────────────────

class BaseDBTest(unittest.TestCase):

    def setUp(self):
        self.db = DatasetManager(":memory:")
        self.frame_ids = []
        self.ann_ids = []
        self._insert_sample_data()

    def tearDown(self):
        self.db.close()

    def _insert_sample_data(self):
        for i, s in enumerate(SAMPLE_ANNOTATIONS):
            frame_id = self.db.add_frame(
                filename=s["filename"],
                relative_path=f"../frame_extractor/extracted_frames/aspave/{s['filename']}",
                width=1920,
                height=1080,
                source="aspave",
                frame_number=i,
            )
            self.frame_ids.append(frame_id)

            ann_id = self.db.add_annotation(
                frame_id=frame_id,
                scene_description=s["scene_description"],
                steering_angle_deg=s["steering_angle_deg"],
                throttle=s["throttle"],
                brake=s["brake"],
                annotation_source=s["annotation_source"],
            )
            self.ann_ids.append(ann_id)

            for category, present in s["labels"].items():
                self.db.add_label_category(ann_id, category, present)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Insert 5 sample annotations
# ─────────────────────────────────────────────────────────────────────────────

class TestInsertAnnotations(BaseDBTest):

    def test_five_frames_registered(self):
        frames = self.db.list_frames()
        self.assertEqual(len(frames), 5)

    def test_five_annotations_stored(self):
        annotations = self.db.get_all_annotations()
        self.assertEqual(len(annotations), 5)

    def test_frame_fields(self):
        frame = self.db.get_frame(self.frame_ids[0])
        self.assertEqual(frame["filename"], "aspave_frame_0000.jpg")
        self.assertEqual(frame["source"], "aspave")
        self.assertEqual(frame["width"], 1920)
        self.assertEqual(frame["height"], 1080)

    def test_annotation_fields(self):
        ann = self.db.get_annotation(self.ann_ids[1])
        self.assertAlmostEqual(ann["steering_angle_deg"], 0.0)
        self.assertAlmostEqual(ann["throttle"], 0.2)
        self.assertAlmostEqual(ann["brake"], 0.5)
        self.assertEqual(ann["annotation_source"], "manual")

    def test_annotation_source_chatgpt(self):
        ann = self.db.get_annotation(self.ann_ids[0])
        self.assertEqual(ann["annotation_source"], "chatgpt")

    def test_labels_stored(self):
        labels = self.db.get_labels_for_annotation(self.ann_ids[1])
        label_map = {l["category"]: bool(l["present"]) for l in labels}
        self.assertTrue(label_map["pedestrian"])
        self.assertFalse(label_map["vehicle"])
        self.assertTrue(label_map["traffic_light"])

    def test_all_five_label_categories_present(self):
        labels = self.db.get_labels_for_annotation(self.ann_ids[0])
        categories = {l["category"] for l in labels}
        self.assertEqual(categories, set(VALID_CATEGORIES))

    def test_duplicate_frame_is_idempotent(self):
        existing_id = self.frame_ids[0]
        returned_id = self.db.add_frame(
            filename="aspave_frame_0000.jpg",
            relative_path="../frame_extractor/extracted_frames/aspave/aspave_frame_0000.jpg",
        )
        self.assertEqual(existing_id, returned_id)
        self.assertEqual(len(self.db.list_frames()), 5)  # no duplicate row

    def test_absolute_path_rejected(self):
        with self.assertRaises(ValueError):
            self.db.add_frame("test.jpg", "/absolute/path/test.jpg")

    def test_timestamps_set(self):
        frame = self.db.get_frame(self.frame_ids[0])
        self.assertIsNotNone(frame["created_at"])
        self.assertIsNotNone(frame["updated_at"])

    def test_get_annotations_for_frame(self):
        anns = self.db.get_annotations_for_frame(self.frame_ids[2])
        self.assertEqual(len(anns), 1)
        self.assertAlmostEqual(anns[0]["steering_angle_deg"], 45.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Query frames by label
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryByLabel(BaseDBTest):

    def test_pedestrian_frames(self):
        results = self.db.get_frames_by_label("pedestrian")
        filenames = {r["filename"] for r in results}
        self.assertIn("aspave_frame_0001.jpg", filenames)
        self.assertIn("aspave_frame_0004.jpg", filenames)
        self.assertNotIn("aspave_frame_0000.jpg", filenames)
        self.assertNotIn("aspave_frame_0002.jpg", filenames)
        self.assertNotIn("aspave_frame_0003.jpg", filenames)

    def test_vehicle_frames(self):
        results = self.db.get_frames_by_label("vehicle")
        filenames = {r["filename"] for r in results}
        self.assertIn("aspave_frame_0000.jpg", filenames)
        self.assertIn("aspave_frame_0002.jpg", filenames)
        self.assertIn("aspave_frame_0004.jpg", filenames)
        self.assertNotIn("aspave_frame_0001.jpg", filenames)
        self.assertNotIn("aspave_frame_0003.jpg", filenames)

    def test_obstacle_frames(self):
        results = self.db.get_frames_by_label("obstacle")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["filename"], "aspave_frame_0003.jpg")

    def test_traffic_light_frames(self):
        results = self.db.get_frames_by_label("traffic_light")
        filenames = {r["filename"] for r in results}
        self.assertIn("aspave_frame_0001.jpg", filenames)
        self.assertIn("aspave_frame_0002.jpg", filenames)

    def test_invalid_category_raises(self):
        with self.assertRaises(ValueError):
            self.db.get_frames_by_label("unknown_thing")

    def test_search_by_description_pedestrian(self):
        results = self.db.search_by_description("pedestrian")
        self.assertEqual(len(results), 2)

    def test_search_by_description_case_insensitive(self):
        results = self.db.search_by_description("HIGHWAY")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["filename"], "aspave_frame_0000.jpg")

    def test_search_by_description_no_match(self):
        results = self.db.search_by_description("unicorn")
        self.assertEqual(len(results), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Export 80/20 train/val split to JSON
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainValSplit(BaseDBTest):

    def test_split_sizes(self):
        train, val = self.db.get_train_val_split(val_ratio=0.2)
        self.assertEqual(len(train) + len(val), 5)
        # 5 * 0.8 = 4.0 → int = 4 train, 1 val
        self.assertEqual(len(train), 4)
        self.assertEqual(len(val), 1)

    def test_no_overlap_between_splits(self):
        train, val = self.db.get_train_val_split(val_ratio=0.2)
        train_ids = {r["annotation_id"] for r in train}
        val_ids = {r["annotation_id"] for r in val}
        self.assertEqual(len(train_ids & val_ids), 0)

    def test_split_is_deterministic(self):
        train1, val1 = self.db.get_train_val_split(seed=42)
        train2, val2 = self.db.get_train_val_split(seed=42)
        self.assertEqual(
            [r["filename"] for r in train1],
            [r["filename"] for r in train2],
        )

    def test_different_seeds_differ(self):
        _, val1 = self.db.get_train_val_split(seed=42)
        _, val2 = self.db.get_train_val_split(seed=99)
        # With 5 records it is possible (but unlikely) they match — just check no crash
        self.assertIsNotNone(val1)
        self.assertIsNotNone(val2)

    def test_export_split_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path, val_path = self.db.export_train_val_split_json(tmpdir)
            self.assertTrue(os.path.exists(train_path))
            self.assertTrue(os.path.exists(val_path))

    def test_export_split_correct_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path, val_path = self.db.export_train_val_split_json(tmpdir)
            with open(train_path) as f:
                train_data = json.load(f)
            with open(val_path) as f:
                val_data = json.load(f)
            self.assertEqual(len(train_data) + len(val_data), 5)

    def test_export_split_json_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path, val_path = self.db.export_train_val_split_json(tmpdir)
            with open(train_path) as f:
                train_data = json.load(f)
            for record in train_data:
                self.assertIn("filename", record)
                self.assertIn("steering_angle_deg", record)
                self.assertIn("throttle", record)
                self.assertIn("brake", record)
                self.assertIn("labels", record)

    def test_export_split_labels_included(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path, _ = self.db.export_train_val_split_json(tmpdir)
            with open(train_path) as f:
                data = json.load(f)
            for record in data:
                self.assertIsInstance(record["labels"], dict)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Verify all steering angles are valid
# ─────────────────────────────────────────────────────────────────────────────

class TestSteeringAngleValidation(BaseDBTest):

    def test_no_invalid_angles_in_sample_data(self):
        invalid = self.db.validate_all_steering_angles()
        self.assertEqual(
            len(invalid), 0,
            f"Invalid steering angles found: {invalid}"
        )

    def test_all_sample_angles_pass_validate_turn_angle(self):
        for ann in self.db.get_all_annotations():
            angle = ann["steering_angle_deg"]
            self.assertTrue(
                validate_turn_angle(angle),
                f"Angle {angle} in {ann['filename']} failed validation"
            )

    def test_boundary_180_accepted(self):
        frame_id = self.db.add_frame("boundary_pos.jpg", "boundary_pos.jpg", frame_number=98)
        ann_id = self.db.add_annotation(frame_id, "Test max", 180.0, 0.5, 0.0)
        ann = self.db.get_annotation(ann_id)
        self.assertAlmostEqual(ann["steering_angle_deg"], 180.0)

    def test_boundary_neg180_accepted(self):
        frame_id = self.db.add_frame("boundary_neg.jpg", "boundary_neg.jpg", frame_number=99)
        ann_id = self.db.add_annotation(frame_id, "Test min", -180.0, 0.5, 0.0)
        ann = self.db.get_annotation(ann_id)
        self.assertAlmostEqual(ann["steering_angle_deg"], -180.0)

    def test_angle_181_rejected(self):
        frame_id = self.db.add_frame("invalid_over.jpg", "invalid_over.jpg", frame_number=100)
        with self.assertRaises(ValueError):
            self.db.add_annotation(frame_id, "Too high", 181.0, 0.5, 0.0)

    def test_angle_neg181_rejected(self):
        frame_id = self.db.add_frame("invalid_under.jpg", "invalid_under.jpg", frame_number=101)
        with self.assertRaises(ValueError):
            self.db.add_annotation(frame_id, "Too low", -181.0, 0.5, 0.0)

    def test_throttle_over1_rejected(self):
        frame_id = self.db.add_frame("throttle_bad.jpg", "throttle_bad.jpg", frame_number=102)
        with self.assertRaises(ValueError):
            self.db.add_annotation(frame_id, "Bad throttle", 0.0, 1.5, 0.0)

    def test_throttle_negative_rejected(self):
        frame_id = self.db.add_frame("throttle_neg.jpg", "throttle_neg.jpg", frame_number=103)
        with self.assertRaises(ValueError):
            self.db.add_annotation(frame_id, "Neg throttle", 0.0, -0.1, 0.0)

    def test_brake_over1_rejected(self):
        frame_id = self.db.add_frame("brake_bad.jpg", "brake_bad.jpg", frame_number=104)
        with self.assertRaises(ValueError):
            self.db.add_annotation(frame_id, "Bad brake", 0.0, 0.5, 2.0)

    def test_brake_negative_rejected(self):
        frame_id = self.db.add_frame("brake_neg.jpg", "brake_neg.jpg", frame_number=105)
        with self.assertRaises(ValueError):
            self.db.add_annotation(frame_id, "Neg brake", 0.0, 0.5, -0.1)

    def test_update_with_invalid_angle_rejected(self):
        with self.assertRaises(ValueError):
            self.db.update_annotation(self.ann_ids[0], steering_angle_deg=999.0)

    def test_update_valid_angle_accepted(self):
        self.db.update_annotation(self.ann_ids[0], steering_angle_deg=15.0)
        ann = self.db.get_annotation(self.ann_ids[0])
        self.assertAlmostEqual(ann["steering_angle_deg"], 15.0)


# ─────────────────────────────────────────────────────────────────────────────
# Additional CRUD / export tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCRUDAndExport(BaseDBTest):

    def test_delete_frame_removes_annotation(self):
        self.db.delete_frame(self.frame_ids[0])
        self.assertIsNone(self.db.get_frame(self.frame_ids[0]))
        self.assertIsNone(self.db.get_annotation(self.ann_ids[0]))

    def test_delete_annotation_removes_labels(self):
        # Deleting the frame cascades to annotations then to label_categories
        self.db.delete_frame(self.frame_ids[0])
        labels = self.db.get_labels_for_annotation(self.ann_ids[0])
        self.assertEqual(len(labels), 0)

    def test_export_to_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            count = self.db.export_to_json(tmp)
            self.assertEqual(count, 5)
            with open(tmp) as f:
                data = json.load(f)
            self.assertEqual(len(data), 5)
            self.assertIn("labels", data[0])
        finally:
            os.unlink(tmp)

    def test_export_to_csv(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as f:
            tmp = f.name
        try:
            count = self.db.export_to_csv(tmp)
            self.assertEqual(count, 5)
            self.assertTrue(os.path.getsize(tmp) > 0)
        finally:
            os.unlink(tmp)

    def test_export_subset_by_frame_ids(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            count = self.db.export_to_json(tmp, frame_ids=self.frame_ids[:2])
            self.assertEqual(count, 2)
        finally:
            os.unlink(tmp)

    def test_get_stats(self):
        stats = self.db.get_stats()
        self.assertEqual(stats["total_frames"], 5)
        self.assertEqual(stats["total_annotations"], 5)
        self.assertIn("chatgpt", stats["by_source"])
        self.assertIn("manual", stats["by_source"])

    def test_relative_paths_only(self):
        for frame in self.db.list_frames():
            self.assertFalse(
                os.path.isabs(frame["relative_path"]),
                f"Absolute path stored: {frame['relative_path']}"
            )

    def test_invalid_label_category_rejected(self):
        with self.assertRaises(ValueError):
            self.db.add_label_category(self.ann_ids[0], "spaceship")


# ─────────────────────────────────────────────────────────────────────────────
# Persistence test (file-based DB)
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = self.tmp.name

    def tearDown(self):
        os.unlink(self.db_path)

    def test_data_survives_reconnect(self):
        with DatasetManager(self.db_path) as db:
            fid = db.add_frame("persist.jpg", "persist.jpg", frame_number=0)
            db.add_annotation(fid, "Persist test", 15.0, 0.4, 0.0)

        with DatasetManager(self.db_path) as db:
            frames = db.list_frames()
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0]["filename"], "persist.jpg")
            anns = db.get_annotations_for_frame(fid)
            self.assertEqual(len(anns), 1)
            self.assertAlmostEqual(anns[0]["steering_angle_deg"], 15.0)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nRUNNING DATABASE UNIT TESTS")
    print("=" * 60)
    unittest.main(verbosity=2)

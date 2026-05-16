#!/usr/bin/env python3
"""
Run YOLO locally on an image folder or route segment and export annotations
without CVAT.

Outputs:
  - YOLO label txt files
  - COCO JSON
  - CVAT-style XML

Default target labels:
  0 pedestrian
  1 vehicle
  2 traffic_light
  3 stop_sign
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def ensure_dependency(import_name, pip_name):
    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"[INFO] Installing missing dependency: {pip_name}")
        subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=True)
        return importlib.import_module(import_name)


Image = ensure_dependency("PIL.Image", "pillow")
YOLO = ensure_dependency("ultralytics", "ultralytics").YOLO


TARGET_LABELS = ["pedestrian", "vehicle", "traffic_light", "stop_sign"]
TARGET_LABEL_TO_ID = {name: idx for idx, name in enumerate(TARGET_LABELS)}

CLASS_MAP = {
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "stop sign": "stop_sign",
    "stop_sign": "stop_sign",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CAMERA_DIRS = ("raw", "raw_front", "raw_left", "raw_right")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate images locally with YOLO and export YOLO/COCO/CVAT files."
    )
    parser.add_argument("segment", help="Segment folder, e.g. datasets/route_3/segment_00")
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Ultralytics model path/name. Default: yolov8s.pt",
    )
    parser.add_argument(
        "--confidence",
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold. Default: 0.5",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold. Default: 0.7",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. 0, cpu, cuda. Default: ultralytics auto-select.",
    )
    parser.add_argument(
        "--no-empty-labels",
        action="store_true",
        help="Do not create empty YOLO txt files for images with no detections.",
    )
    return parser.parse_args()


def images_in_dir(image_dir):
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def discover_image_sets(folder):
    """Return [(camera_name, image_dir, images)] for an image dir or segment dir."""
    input_dir = Path(folder).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Folder not found: {input_dir}")

    direct_images = images_in_dir(input_dir)
    if direct_images:
        camera_name = input_dir.name if input_dir.name in CAMERA_DIRS else "images"
        return input_dir, [(camera_name, input_dir, direct_images)], False

    image_sets = []
    for camera_name in CAMERA_DIRS:
        camera_dir = input_dir / camera_name
        if not camera_dir.is_dir():
            continue
        images = images_in_dir(camera_dir)
        if images:
            image_sets.append((camera_name, camera_dir, images))

    if not image_sets:
        raise FileNotFoundError(
            f"No images found in {input_dir} or camera folders: {', '.join(CAMERA_DIRS)}"
        )

    return input_dir, image_sets, True


def make_output_dirs(base_dir, camera_name, multi_camera):
    if multi_camera:
        root_dir = base_dir / "annotations"
    else:
        root_dir = base_dir.parent / "annotations"

    out_dir = root_dir / camera_name if multi_camera else root_dir
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir, out_dir, labels_dir


def list_images(folder):
    image_dir = Path(folder).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    images = images_in_dir(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    return image_dir, images


def make_output_dir(image_dir, output_dir):
    if output_dir:
        out_dir = Path(output_dir).expanduser().resolve()
    else:
        out_dir = image_dir.parent / "annotations"

    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, labels_dir


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def detect_image(model, image_path, confidence, iou, device):
    kwargs = {
        "source": str(image_path),
        "conf": confidence,
        "iou": iou,
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device

    results = model.predict(**kwargs)
    if not results:
        return []

    result = results[0]
    if result.boxes is None:
        return []

    detections = []
    names = result.names

    with Image.open(image_path) as img:
        width, height = img.size

    for box in result.boxes:
        raw_class_id = int(box.cls.item())
        raw_label = names.get(raw_class_id, str(raw_class_id))
        target_label = CLASS_MAP.get(raw_label)
        if target_label is None:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = clamp(float(x1), 0.0, float(width))
        y1 = clamp(float(y1), 0.0, float(height))
        x2 = clamp(float(x2), 0.0, float(width))
        y2 = clamp(float(y2), 0.0, float(height))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "label": target_label,
                "class_id": TARGET_LABEL_TO_ID[target_label],
                "confidence": float(box.conf.item()),
                "bbox_xyxy": [x1, y1, x2, y2],
                "width": width,
                "height": height,
            }
        )

    return detections


def write_yolo_label(label_path, detections):
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        width = det["width"]
        height = det["height"]

        box_w = x2 - x1
        box_h = y2 - y1
        x_center = x1 + box_w / 2.0
        y_center = y1 + box_h / 2.0

        lines.append(
            f"{det['class_id']} "
            f"{x_center / width:.8f} "
            f"{y_center / height:.8f} "
            f"{box_w / width:.8f} "
            f"{box_h / height:.8f}"
        )

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_coco(images, detections_by_image):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": idx, "name": name, "supercategory": "object"}
            for idx, name in enumerate(TARGET_LABELS)
        ],
    }

    ann_id = 1
    for image_id, image_path in enumerate(images, start=1):
        detections = detections_by_image[image_path]
        if detections:
            width = detections[0]["width"]
            height = detections[0]["height"]
        else:
            with Image.open(image_path) as img:
                width, height = img.size

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            box_w = x2 - x1
            box_h = y2 - y1
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": det["class_id"],
                    "bbox": [x1, y1, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                    "score": det["confidence"],
                }
            )
            ann_id += 1

    return coco


def build_cvat_xml(images, detections_by_image):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "annotations"
    labels_node = ET.SubElement(task, "labels")
    for label_name in TARGET_LABELS:
        label_node = ET.SubElement(labels_node, "label")
        ET.SubElement(label_node, "name").text = label_name
        ET.SubElement(label_node, "color").text = "#000000"
        ET.SubElement(label_node, "type").text = "rectangle"
        ET.SubElement(label_node, "attributes")

    for image_id, image_path in enumerate(images):
        detections = detections_by_image[image_path]
        if detections:
            width = detections[0]["width"]
            height = detections[0]["height"]
        else:
            with Image.open(image_path) as img:
                width, height = img.size

        image_node = ET.SubElement(
            root,
            "image",
            {
                "id": str(image_id),
                "name": image_path.name,
                "width": str(width),
                "height": str(height),
            },
        )

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            ET.SubElement(
                image_node,
                "box",
                {
                    "label": det["label"],
                    "source": "auto",
                    "occluded": "0",
                    "xtl": f"{x1:.2f}",
                    "ytl": f"{y1:.2f}",
                    "xbr": f"{x2:.2f}",
                    "ybr": f"{y2:.2f}",
                    "z_order": "0",
                },
            )

    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def write_labels_file(output_dir):
    labels_path = output_dir / "classes.txt"
    labels_path.write_text("\n".join(TARGET_LABELS) + "\n", encoding="utf-8")


def annotate_image_set(args, model, camera_name, image_dir, images, output_dir, labels_dir):
    detections_by_image = {}
    total_boxes = 0

    print(f"\n[*] Camera: {camera_name}")
    print(f"[*] Found {len(images)} images in {image_dir}")
    print(f"[*] Writing annotations to {output_dir}")

    for index, image_path in enumerate(images, start=1):
        detections = detect_image(
            model=model,
            image_path=image_path,
            confidence=args.confidence,
            iou=args.iou,
            device=args.device,
        )
        detections_by_image[image_path] = detections
        total_boxes += len(detections)

        label_path = labels_dir / f"{image_path.stem}.txt"
        if detections or not args.no_empty_labels:
            write_yolo_label(label_path, detections)

        print(f"[{index:04d}/{len(images):04d}] {image_path.name}: {len(detections)} boxes")

    coco = build_coco(images, detections_by_image)
    coco_path = output_dir / "annotations_coco.json"
    coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    cvat_xml_path = output_dir / "annotations_cvat.xml"
    cvat_tree = build_cvat_xml(images, detections_by_image)
    cvat_tree.write(cvat_xml_path, encoding="utf-8", xml_declaration=True)

    summary = {
        "camera": camera_name,
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "confidence": args.confidence,
        "images": len(images),
        "boxes": total_boxes,
        "labels": TARGET_LABELS,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "camera": camera_name,
        "images": len(images),
        "boxes": total_boxes,
        "labels_dir": str(labels_dir),
        "coco_path": str(coco_path),
        "cvat_xml_path": str(cvat_xml_path),
    }


def annotate_target(args, model, target):
    base_dir, image_sets, multi_camera = discover_image_sets(target)
    all_summaries = []
    root_output_dir = None

    for camera_name, image_dir, images in image_sets:
        root_output_dir, output_dir, labels_dir = make_output_dirs(
            base_dir=base_dir,
            camera_name=camera_name,
            multi_camera=multi_camera,
        )
        write_labels_file(output_dir)
        if multi_camera:
            write_labels_file(root_output_dir)
        all_summaries.append(
            annotate_image_set(args, model, camera_name, image_dir, images, output_dir, labels_dir)
        )

    if root_output_dir is not None:
        (root_output_dir / "summary.json").write_text(
            json.dumps({"input_dir": str(base_dir), "multi_camera": multi_camera, "cameras": all_summaries}, indent=2),
            encoding="utf-8",
        )

    return root_output_dir, all_summaries


def discover_annotation_targets(target):
    input_dir = Path(target).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Folder not found: {input_dir}")

    direct_images = images_in_dir(input_dir)
    camera_has_images = any(
        (input_dir / camera_name).is_dir() and images_in_dir(input_dir / camera_name)
        for camera_name in CAMERA_DIRS
    )
    if direct_images or camera_has_images:
        return [input_dir]

    segments = sorted(
        path for path in input_dir.iterdir()
        if path.is_dir() and path.name.startswith("segment_")
    )
    if segments:
        return segments

    return [input_dir]


def main():
    args = parse_args()
    targets = discover_annotation_targets(args.segment)

    print(f"[*] Loading model: {args.model}")
    model = YOLO(args.model)

    all_summaries = []
    output_dirs = []
    for target in targets:
        print(f"\n[*] Annotating target: {target}")
        root_output_dir, target_summaries = annotate_target(args, model, target)
        if root_output_dir is not None:
            output_dirs.append(root_output_dir)
        all_summaries.extend(target_summaries)

    total_images = sum(item["images"] for item in all_summaries)
    total_boxes = sum(item["boxes"] for item in all_summaries)

    print("\n[SUCCESS] Local annotation complete")
    print(f"Cameras:      {', '.join(item['camera'] for item in all_summaries)}")
    print(f"Images:       {total_images}")
    print(f"Boxes:        {total_boxes}")
    print("Output:")
    for output_dir in output_dirs:
        print(f"  - {output_dir}")


if __name__ == "__main__":
    main()

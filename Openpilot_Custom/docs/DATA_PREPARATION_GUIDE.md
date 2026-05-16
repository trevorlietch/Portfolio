# Openpilot Dataset Prep + Offline Labeling Guide (Replay → Dataset for Later Training)

This document describes a student-friendly workflow to turn an **openpilot replay** into a structured dataset that is ready for later training and analysis. The core idea is simple: **for each frame**, save the camera image and a matching model feature file, then (optionally) generate offline labels like depth maps and object detections using standard CV tools.

---

## ✅ What you will produce (deliverables)

You will create an output directory containing **multiple segments** (for example, `segment_00 … segment_11`). Each segment contains frame-aligned data:

- **`raw/`**: camera images (`000000.png`, `000001.png`, …)
- **`features/`**: model feature embeddings saved per frame (same indexing)
- **Optional labels (offline)**:
  - **`depth_npy/`**: per-frame depth map arrays (`000000.npy`, …)
  - **`labels/`**: per-frame detection labels (YOLO `.txt` or JSON)
  - **Merged labels**: one record per frame combining detections + depth statistics

---

## ✅ SYSTEM REQUIREMENTS

| Requirement | Notes |
|---|---|
| Linux + Python 3 | Use the same environment you use for openpilot tools |
| `openpilot` repo | You will modify one file or add one capture script |
| Enough disk space | Datasets can become large quickly |
| GPU (recommended) | Greatly speeds up depth + detection labeling |

---

## Data preparation steps (Replay → Captured Dataset)

### STEP 1 — Replay a multi-segment route
Use openpilot’s replay tooling to play back a route with multiple segments. This recreates the message streams openpilot would see in real driving (camera frames + vehicle/state signals).

**Output of this step:** a running replay that publishes camera frames and state messages.

### STEP 2 — Capture training inputs during replay (images + features)
Run a dataset capture process during replay (typically by modifying/using a `modeld` variant). The capture logic should:

- read road camera frames from openpilot’s camera stream
- save the **raw image** to disk
- run openpilot’s vision model and save a per-frame **feature embedding** (“hidden_state” / model feature vector)
- write into a segment-based folder structure (e.g., `segment_00`, `segment_01`, …)

**Why features?** Many learning pipelines train a policy on these embeddings (or use them as additional inputs/metadata), so saving them with exact alignment to frames is crucial.

### STEP 3 — Organize outputs by segment and frame index
Ensure each saved file uses a consistent, numeric index:

- `raw/000123.png`
- `features/000123.*` (same index)

This makes future joining trivial (image ↔ features ↔ labels).

### STEP 4 — Quick verification (counts + spot-check)
Before doing any labeling, verify that each segment’s `raw/` and `features/` counts match and that images are readable.

---

## Offline labeling steps (Images → Depth + Detection Labels)

Offline labeling is done **after** capture to keep replay/capture simple and to make experimentation fast.

### STEP 5 — Depth estimation (per frame)
Run a depth estimation model on each `raw/*.png` to generate a per-pixel **depth map**.

- Save depth to a machine-readable format (commonly `.npy`) using the same frame index:
  - `raw/000123.png → depth_npy/000123.npy`
- Depth can be **relative depth** (acceptable for this project) unless metric calibration is explicitly required.

### STEP 6 — Object detection (per frame)
Run an object detector (e.g., YOLO) on each `raw/*.png` to produce:

- bounding boxes
- class labels
- confidence scores

Save detections aligned by frame index:
- `raw/000123.png → labels/000123.txt` (YOLO format) or `000123.json`

### STEP 7 — Merge depth + detection (optional, recommended)
To attach an approximate “near/far” value to each detected object:

- load the depth map (`depth_npy/000123.npy`)
- for each detection box, compute depth statistics inside the box (median/mean)
- write one merged record per frame (or a single JSONL/CSV over all frames)

**Note:** detector confidence is **not distance**. Distance-like signals come from depth (or from a separate range sensor / calibration).

---

## Quick checks

- **Index alignment**: `000123.png` should have matching `000123` in features/depth/labels where applicable.
- **Counts**: `raw` and `features` should match exactly; depth should match `raw`; detection labels may be fewer if a frame has no detections.
- **Spot-check**: open a few frames and verify depth/detections look reasonable.

---

## Troubleshooting quick table

| Symptom | Likely cause | Fix |
|---|---|---|
| Only one segment was saved | No segmentation rule or replay didn’t expose segment boundaries | Use a frame-count-based segmentation rule (e.g., new segment every N frames) |
| Missing detections for some frames | No objects detected | This can be normal; keep a consistent merged format |
| Depth looks “wrong” | Depth is relative / normalization differs | Validate qualitatively; metric depth requires extra calibration |
| Dataset too large | Capturing too many frames | Reduce capture duration or increase segment size and limit segments |

---

## ✔ Summary (what to submit)

| Item | Description |
|---|---|
| Captured dataset | `segment_XX/raw` + `segment_XX/features` |
| Optional depth labels | `segment_XX/depth_npy` |
| Optional detection labels | `segment_XX/labels` (YOLO or JSON) |
| Optional merged labels | One merged file per segment or one JSONL/CSV for the full dataset |
| Short report | How many segments/frames, and example visualizations |


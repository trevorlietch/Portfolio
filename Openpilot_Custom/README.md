# Openpilot Dataset Preparation Project

This repository contains the **modified openpilot files** and **guide** you need to complete a dataset preparation project. Your goal is to turn an openpilot driving replay into a structured dataset with images, model features, depth maps, and object detections.

---

## 📋 What This Repository Contains

This repo contains **ONLY**:

- **`docs/DATA_PREPARATION_GUIDE.md`** — Step-by-step guide explaining what you need to do
- **`openpilot_files/selfdrive/modeld_detection_first.py`** — Modified modeld file (copy this into your openpilot)
- **`openpilot_files/selfdrive/modeld_detection_second.py`** — Alternative modified modeld file (copy this into your openpilot)

**You will need to provide:**
- Openpilot repository (v0.9.8)
- YOLO (for object detection)
- Depth Anything V2 (for depth estimation)

---

## 🚀 Quick Start

### Step 1: Run the automated setup script

Please refer to the **[main README](../../README.md)** in the root of `project19`. 

We have provided a `setup_openpilot.sh` script that handles:
- Cloning openpilot
- Installing dependencies (`ubuntu_setup.sh`)
- Building openpilot (`scons`)
- Copying the custom modeld files from this folder

**Do not follow the manual steps below unless the script fails.**

**Note:** You can use either `modeld_detection_first.py` or `modeld_detection_second.py`. The second version includes automatic segment management (saves to `segment_00`, `segment_01`, etc.).

### Step 4: Route Play
You may use the following public routes for testing:

- `d34c14daa88a1e86/0000013e--0859dd3dcc`
- `d34c14daa88a1e86/000000ca--7c5d326170`

Run replay using:

tools/replay/replay d34c14daa88a1e86/0000013e--0859dd3dcc


### Step 5: Follow the guide

Read **`docs/DATA_PREPARATION_GUIDE.md`** for the complete workflow:

1. **Replay a route** using openpilot's replay tool
2. **Capture images + features** using the modified modeld file
3. **Run offline labeling:**
   - Use **YOLO** to detect objects (cars, pedestrians, etc.)
   - Use **Depth Anything V2** to estimate depth maps
---


## 📖 Full Instructions

See **`docs/DATA_PREPARATION_GUIDE.md`** for the complete step-by-step guide.

---

## ❓ Troubleshooting

If you encounter issues:

1. Make sure your openpilot environment is set up correctly
2. Verify the modified modeld file is in the right location
3. Check that replay is running and publishing camera frames
4. See the troubleshooting section in `docs/DATA_PREPARATION_GUIDE.md`



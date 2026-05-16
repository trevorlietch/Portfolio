import os
import sys
import glob
import argparse
import json

def get_default_route():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    dirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    if dirs:
        dirs.sort()
        return dirs[0]
    return None

parser = argparse.ArgumentParser(description="Headless Alpamayo batch inference JSON exporter.")
parser.add_argument("--route", type=str, default=get_default_route(), 
                    help="Path to a route directory containing segments")
parser.add_argument("--frames", type=int, default=64, 
                    help="Number of future frames to graph for predictions")
parser.add_argument("--segment", type=str, default=None,
                    help="Process only a specific segment (e.g., 'segment_00')")
parser.add_argument("--start-frame", type=int, default=0,
                    help="First frame index to process within the segment (inclusive)")
parser.add_argument("--end-frame", type=int, default=None,
                    help="Last frame index to process within the segment (inclusive)")
parser.add_argument("--num-traj-samples", type=int, default=16,
                    help="Number of trajectory samples to draw per condition")
parser.add_argument("--selection-mode", choices=["heuristic", "mean", "median"], default="heuristic",
                    help="How to collapse sampled trajectories into the displayed path.")
parser.add_argument("--guidance-weight", type=float, default=1.5,
                    help="Classifier-free guidance weight for nav-conditioned inference")
parser.add_argument("--cameras", nargs="+", choices=["wide", "left", "right", "front"],
                    default=["wide", "left", "right", "front"],
                    help="Cameras to include (wide, left, right, front). Unlisted cameras will be excluded.")
parser.add_argument("--max-gen-length", type=int, default=256,
                    help="Maximum generation length for the trajectory diffusion model. Lower speeds it up but reduces max distance.")
parser.add_argument("--plot-all-samples", action="store_true",
                    help="Deprecated compatibility flag. Videos are rendered later from saved prediction JSON.")
global_args = parser.parse_args()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
import signal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from alpamayo1_5.load_custom_dataset import load_custom_dataset
from alpamayo1_5.navigation_command import infer_navigation_command
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


def extract_cot(extra, idx=0):
    try:
        # Alpamayo wraps extra["cot"] in nested dimensions [batch=1, beam=1, samples=16]
        cot_data = extra.get("cot", [])
        
        # Unwrap extraneous [1] outer dimensions until we hit the actual array of 16 options
        while isinstance(cot_data, (list, tuple, np.ndarray)) and len(cot_data) == 1:
            cot_data = cot_data[0]
            
        # Select the exact string that matches the statistically chosen trajectory index
        if isinstance(cot_data, (list, tuple, np.ndarray)):
            if len(cot_data) > idx:
                return str(cot_data[idx]).strip()
            elif len(cot_data) > 0:
                return str(cot_data[0]).strip()
        
        return str(cot_data).strip()
    except Exception:
        return ""


def run_nav_inference(
    model,
    processor,
    data,
    device,
    nav_cmd: str,
    num_traj_samples: int,
    guidance_weight: float,
    max_gen_length: int = 256,
):
    messages_nav = helper.create_message(
        data["image_frames"].flatten(0, 1),
        camera_indices=data.get("camera_indices"),
        nav_text=nav_cmd,
    )
    inputs_nav = processor.apply_chat_template(
        messages_nav,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs_nav = helper.to_device(
        {
            "tokenized_data": inputs_nav,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        device,
    )

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        pred_xyz_nav, pred_rot_nav, extra_nav = (
            model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav(
                data=model_inputs_nav,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=num_traj_samples,
                max_generation_length=max_gen_length,
                return_extra=True,
                diffusion_kwargs={
                    "use_classifier_free_guidance": True,
                    "inference_guidance_weight": guidance_weight,
                    "temperature": 0.6,
                },
            )
        )

    return pred_xyz_nav, pred_rot_nav, extra_nav


def select_prediction_path(
    pred_tensor,
    nav_cmd: str,
    num_frames: int,
    selection_mode: str = "heuristic",
) -> tuple[np.ndarray, int, int, np.ndarray]:
    pred_np = pred_tensor.detach().cpu().numpy()[0, 0]
    if pred_np.shape[0] == 0:
        return np.zeros((1, 3), dtype=np.float32), 0, 0, pred_np

    if selection_mode == "mean":
        selected = pred_np.mean(axis=0)
        sample_idx = int(
            np.argmin(
                np.linalg.norm(pred_np[:, :, :2] - selected[None, :, :2], axis=-1).mean(axis=1)
            )
        )
    elif selection_mode == "median":
        selected = np.median(pred_np, axis=0)
        sample_idx = int(
            np.argmin(
                np.linalg.norm(pred_np[:, :, :2] - selected[None, :, :2], axis=-1).mean(axis=1)
            )
        )
    else:
        nav_lower = nav_cmd.lower()
        final_lateral = pred_np[:, -1, 1]
        final_forward = pred_np[:, -1, 0]

        if "left" in nav_lower:
            sample_idx = int(np.argmax(final_lateral))
        elif "right" in nav_lower:
            sample_idx = int(np.argmin(final_lateral))
        elif "straight" in nav_lower:
            sample_idx = int(np.argmin(np.abs(final_lateral)))
        else:
            # Fallback to the sample that goes furthest forward while staying centered.
            sample_idx = int(np.argmax(final_forward - np.abs(final_lateral)))

        selected = pred_np[sample_idx]

    n_frames = min(num_frames, selected.shape[0])
    return selected[:n_frames], n_frames, sample_idx, pred_np


def path_to_records(path: np.ndarray) -> list[dict]:
    records = []
    for idx, point in enumerate(path):
        records.append(
            {
                "step_index": int(idx),
                "x_m": float(point[0]),
                "y_m": float(point[1]),
                "z_m": float(point[2]) if len(point) > 2 else 0.0,
            }
        )
    return records


def samples_to_records(samples: np.ndarray, num_frames: int) -> list[dict]:
    return [
        {
            "sample_index": int(sample_idx),
            "path": path_to_records(sample[:num_frames]),
        }
        for sample_idx, sample in enumerate(samples)
    ]


def prediction_json_dir(args, seg_dir: str) -> str:
    return os.path.join(seg_dir, "predictions")


def save_prediction_json(
    args,
    route_name: str,
    seg_name: str,
    seg_dir: str,
    local_idx: int,
    nav_cmd: str,
    cmd_text: str,
    cot: str,
    sample_idx: int,
    selected_path: np.ndarray,
    gt_xyz: np.ndarray,
    n_frames: int,
    data: dict,
) -> str:
    out_dir = prediction_json_dir(args, seg_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{seg_name}_{local_idx:06d}_prediction.json")

    payload = {
        "schema_version": 1,
        "model_name": "nvidia/Alpamayo-1.5-10B",
        "route": route_name,
        "segment": seg_name,
        "frame_index": int(local_idx),
        "clip_id": data.get("clip_id"),
        "t0_us": data.get("t0_us"),
        "nav_command": nav_cmd,
        "command": cmd_text,
        "command_text": cmd_text,
        "command_source": "ground_truth_heuristic",
        "selection_mode": args.selection_mode,
        "selected_sample_index": int(sample_idx),
        "num_traj_samples": int(args.num_traj_samples),
        "guidance_weight": float(args.guidance_weight),
        "max_generation_length": int(args.max_gen_length),
        "frames_requested": int(args.frames),
        "frames_stored": int(n_frames),
        "cameras": args.cameras,
        "reasoning_text": cot,
        "reasoning": cot,
        "selected_path": path_to_records(selected_path),
        "ground_truth_path": path_to_records(gt_xyz[:n_frames]),
    }

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return out_path


def main():
    args = global_args
    
    # Safe interrupt flag to gracefully finish the current frame.
    interrupt_flag = [False]
    def graceful_exit(sig, frame):
        print("\n\n[Stop Signal Detected] Finishing the current frame, please wait...")
        interrupt_flag[0] = True
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    if not args.route or not os.path.exists(args.route):
        print(f"Error: Route directory '{args.route}' not found.")
        return

    cam_mapping = {"left": 0, "wide": 1, "right": 2, "front": 6}
    excluded_cameras = []
    for name, idx in cam_mapping.items():
        if name not in args.cameras:
            excluded_cameras.append(idx)

    print("Loading Alpamayo model... (This will take a moment)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", 
        dtype=torch.bfloat16,
        attn_implementation="eager").to(device)
    if device == "cuda":
        print("Compiling model for faster inference (this may take a few minutes on the first run)...")
        model = torch.compile(model)
    
    processor = helper.get_processor(model.tokenizer)

    all_segments = sorted([d for d in glob.glob(os.path.join(args.route, 'segment_*')) if os.path.isdir(d)])
    if args.segment:
        all_segments = [s for s in all_segments if os.path.basename(s) == args.segment]

    if not all_segments:
        print("No segments found in route.")
        return

    print(f"Found {len(all_segments)} segments in {args.route}.")

    for seg_dir in all_segments:
        seg_name = os.path.basename(seg_dir)
        print(f"\nProcessing {seg_name}...")
        
        telemetry_dir = os.path.join(seg_dir, "telemetry")
        
        # raw folder is front wide, raw_front is front narrow
        raw_dir = os.path.join(seg_dir, "raw_front")
        if not os.path.exists(raw_dir):
            raw_dir = os.path.join(seg_dir, "raw")
            
        if not os.path.exists(telemetry_dir) or not os.path.exists(raw_dir):
            print(f"Skipping {seg_name} due to missing data.")
            continue
            
        num_frames_seg = len(glob.glob(os.path.join(telemetry_dir, "*.json")))
        if num_frames_seg == 0:
            print(f"Skipping {seg_name} because it has no telemetry frames.")
            continue

        start_frame = args.start_frame
        end_frame = num_frames_seg - 1 if args.end_frame is None else args.end_frame

        if start_frame < 0:
            print(f"Skipping {seg_name}: start frame {start_frame} must be >= 0.")
            continue
        if end_frame < 0:
            print(f"Skipping {seg_name}: end frame {end_frame} must be >= 0.")
            continue
        if start_frame >= num_frames_seg:
            print(
                f"Skipping {seg_name}: start frame {start_frame} is outside the segment "
                f"(0-{num_frames_seg - 1})."
            )
            continue
        if end_frame >= num_frames_seg:
            print(
                f"Skipping {seg_name}: end frame {end_frame} is outside the segment "
                f"(0-{num_frames_seg - 1})."
            )
            continue
        if end_frame < start_frame:
            print(
                f"Skipping {seg_name}: end frame {end_frame} is before start frame {start_frame}."
            )
            continue
        
        print(
            f"Processing frames {start_frame} through {end_frame} "
            f"(inclusive) out of 0-{num_frames_seg - 1}."
        )

        route_name = os.path.basename(os.path.abspath(args.route))
        for local_idx in range(start_frame, end_frame + 1):
            if interrupt_flag[0]:
                break

            try:
                data = load_custom_dataset(seg_dir, local_idx, exclude_cameras=excluded_cameras)
            except Exception as e:
                print(f"Error loading data for {seg_name} frame {local_idx}: {e}")
                continue

            gt_xyz = data["ego_future_xyz"][0, 0].numpy()
            
            nav_cmd = infer_navigation_command(gt_xyz)

            # Set fixed seed to match the nav notebook exactly for deterministic conditional inference
            torch.cuda.manual_seed_all(42)

            cot = ""
            pred_xyz_nav, _, extra_nav = run_nav_inference(
                model=model,
                processor=processor,
                data=data,
                device=device,
                nav_cmd=nav_cmd,
                num_traj_samples=args.num_traj_samples,
                guidance_weight=args.guidance_weight,
                max_gen_length=args.max_gen_length,
            )

            for cmd_text, pred_xyz, extra in [(nav_cmd, pred_xyz_nav, extra_nav)]:
                selected_path, selected_frames, sample_idx, _ = select_prediction_path(
                    pred_xyz,
                    cmd_text,
                    args.frames,
                    selection_mode=args.selection_mode,
                )
                
                cot = extract_cot(extra, sample_idx)
                if args.selection_mode == "heuristic":
                    print(
                        f"[{seg_name} | Frame {local_idx}] Cmd: \033[92m{cmd_text}\033[0m | "
                        f"Reasoning: \033[38;2;255;165;0m{cot}\033[0m"
                    )
                else:
                    print(
                        f"[{seg_name} | Frame {local_idx}] Cmd: \033[92m{cmd_text}\033[0m | "
                        f"Display Path: \033[96m{args.selection_mode}\033[0m | "
                        f"Representative Reasoning: \033[38;2;255;165;0m{cot}\033[0m"
                    )

                save_prediction_json(
                    args=args,
                    route_name=route_name,
                    seg_name=seg_name,
                    seg_dir=seg_dir,
                    local_idx=local_idx,
                    nav_cmd=nav_cmd,
                    cmd_text=cmd_text,
                    cot=cot,
                    sample_idx=sample_idx,
                    selected_path=selected_path,
                    gt_xyz=gt_xyz,
                    n_frames=selected_frames,
                    data=data,
                )
        print(f"Finished writing prediction JSON for {seg_name}.")
        
        if interrupt_flag[0]:
            print("Processing stopped early by user.")
            break

if __name__ == "__main__":
    main()

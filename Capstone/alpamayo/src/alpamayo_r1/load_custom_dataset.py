# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom dataset loader for Alpamayo R1

import os
import json
import torch
import numpy as np
from PIL import Image

def load_custom_dataset(
    segment_dir: str,
    frame_idx: int,
    num_history_steps: int = 16, # Kinematic history length at 10Hz
    num_future_steps: int = 64,  # GT future length at 10Hz
    time_step: float = 0.1,      # Target framerate matching Alpamayo (10Hz)
    frame_stride: int = 2,       # Source dataset stride (Openpilot is 20Hz, so stride 2 = 10Hz)
):
    """
    Loads custom openpilot-style data and converts to Alpamayo format.
    Expects segment_dir to contain:
      - raw/ (containing .png images named like 000000.png)
      - telemetry/ (containing .json files named like 000000.json with v_ego, yaw_rate, steering_angle_deg)
    """
    
    raw_dir = os.path.join(segment_dir, "raw")
    telemetry_dir = os.path.join(segment_dir, "telemetry")
    
    # 1. Kinematic Integration (History & Future)
    # start at t0 (frame_idx) as Origin (0,0,0, identity rot).
    
    # History Integration (Backwards from t0)
    hist_xyz = []
    hist_rot = []
    hist_xyz.append(np.zeros(3)) # t0 is 0,0,0
    hist_rot.append(np.eye(3))   # t0 is identity
    
    x, y, theta = 0.0, 0.0, 0.0
    
    # Current State at t0
    current_json = os.path.join(telemetry_dir, f"{frame_idx:06d}.json")
    t0_us = 0
    if os.path.exists(current_json):
        with open(current_json, 'r') as f:
            d = json.load(f)
            t0_us = d.get('timestamp_eof', 0) / 1000 # convert ns to us if needed based on your telemetry
    
    # Iterate backwards
    for i in range(1, num_history_steps):
        prev_idx = frame_idx - (i * frame_stride)
        if prev_idx < 0: prev_idx = 0 # Clamp to start
        
        json_path = os.path.join(telemetry_dir, f"{prev_idx:06d}.json")
        v = 0.0
        w = 0.0
        dt = time_step
        is_reverse = False
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                v = data.get('v_ego', 0.0)
                yaw_rate = data.get('yaw_rate', 0.0)
                steer_deg = data.get('steering_angle_deg', 0.0)
                is_reverse = data.get('gear_shifter') == 'reverse'
                
                # Dynamic dt calculation
                t_prev_us = data.get('timestamp_eof', 0) / 1000
                if t0_us > 0 and t_prev_us > 0:
                    dt = (t0_us - t_prev_us) / 1000000.0 / i # approximate avg dt
                
                
                # Yaw Rate Fallback (Bicycle Model)
                if abs(yaw_rate) < 1e-4 and abs(steer_deg) > 0.5:
                     steer_rad = np.deg2rad(steer_deg) / 15.49
                     yaw_rate = v * np.tan(steer_rad) / 2.7
                
                w = yaw_rate
        
        # Integrate Backwards
        if is_reverse:
             x -= (-v) * np.cos(theta) * dt
             y -= (-v) * np.sin(theta) * dt
             theta -= (-w) * dt
        else:
             x -= v * np.cos(theta) * dt
             y -= v * np.sin(theta) * dt
             theta -= w * dt
        
        hist_xyz.append(np.array([x, y, 0.0]))
        
        # Rotation Matrix from theta
        c, s = np.cos(theta), np.sin(theta)
        R_mat = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        hist_rot.append(R_mat)
        
    # Reverse to be chronological [t-N, ..., t0]
    hist_xyz = hist_xyz[::-1]
    hist_rot = hist_rot[::-1]
    
    # Future Integration (Forwards from t0)
    fut_xyz = []
    fut_rot = []
    
    x, y, theta = 0.0, 0.0, 0.0 # Reset to t0
    
    for i in range(1, num_future_steps + 1):
        next_idx = frame_idx + (i * frame_stride)
        json_path = os.path.join(telemetry_dir, f"{next_idx:06d}.json")
        v = 0.0
        w = 0.0
        dt = time_step
        is_reverse = False
        
        if os.path.exists(json_path):
             with open(json_path, 'r') as f:
                data = json.load(f)
                v = data.get('v_ego', 0.0)
                yaw_rate = data.get('yaw_rate', 0.0)
                steer_deg = data.get('steering_angle_deg', 0.0)
                is_reverse = data.get('gear_shifter') == 'reverse'
                
                # Dynamic dt calculation
                t_next_us = data.get('timestamp_eof', 0) / 1000
                if t0_us > 0 and t_next_us > 0:
                    dt = (t_next_us - t0_us) / 1000000.0 / i # approximate avg dt

                if abs(yaw_rate) < 1e-4 and abs(steer_deg) > 0.5:
                     steer_rad = np.deg2rad(steer_deg) / 15.49
                     yaw_rate = v * np.tan(steer_rad) / 2.7
                w = yaw_rate
                
        if is_reverse:
             x += (-v) * np.cos(theta) * dt
             y += (-v) * np.sin(theta) * dt
             theta += (-w) * dt
        else:
             x += v * np.cos(theta) * dt
             y += v * np.sin(theta) * dt
             theta += w * dt
        
        fut_xyz.append(np.array([x, y, 0.0]))
        c, s = np.cos(theta), np.sin(theta)
        R_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        fut_rot.append(R_mat)
        
    # Convert to Tensors
    # Shape: (1, 1, Steps, 3)
    ego_history_xyz = torch.tensor(np.stack(hist_xyz), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ego_history_rot = torch.tensor(np.stack(hist_rot), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ego_future_xyz = torch.tensor(np.stack(fut_xyz), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ego_future_rot = torch.tensor(np.stack(fut_rot), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 3. Load Images
    # Alpamayo expects (N_cameras, num_frames, 3, H, W)
    
    num_visual_frames = 4 # Default in load_physical_aiavdataset
    images = []
    
    # Indices for visual frames
    for i in range(num_visual_frames):
        # Index: frame_idx - (3 - i) * stride
        idx = frame_idx - (num_visual_frames - 1 - i) * frame_stride
        if idx < 0: idx = 0
        img_path = os.path.join(raw_dir, f"{idx:06d}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            # You can optionally resize here if needed, eg. img.resize((W, H))
            img_np = np.array(img)
        else:
            # If image missing, pad with zeros or a previous image
            img_np = np.zeros((224, 224, 3), dtype=np.uint8) # Placeholder
            
        images.append(img_np)
        
    # Stack: (num_frames, H, W, 3) -> (num_frames, 3, H, W)
    images_tensor = torch.tensor(np.stack(images), dtype=torch.uint8).permute(0, 3, 1, 2)
    
    # Wrap in Camera Dimension
    # Alpamayo rigidly expects 4 cameras (16 images total) in the order: [Cross Left, Front Wide, Cross Right, Front Tele]
    # We only have one camera (Front Wide), so we pad the missing perspectives with black frames to ensure 
    # the front camera aligns with the correct visual tokens in the prompt.
    front_wide_tensor = images_tensor.unsqueeze(0) # (1, num_frames, 3, H, W)
    black_tensor = torch.zeros_like(front_wide_tensor)
    
    image_frames = torch.cat([
        black_tensor,       # 0: Cross Left (Index 0)
        front_wide_tensor,  # 1: Front Wide (Index 1)
        black_tensor,       # 2: Cross Right (Index 2)
        front_wide_tensor,  # 3: Front Tele (Index 6, copied from front wide)
    ], dim=0) # (4, num_frames, 3, H, W)
    
    camera_indices = torch.tensor([0, 1, 2, 6], dtype=torch.int64)
    
    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "ego_future_xyz": ego_future_xyz,  # useful for eval, omit if you don't have future telemetry
        "ego_future_rot": ego_future_rot,  # useful for eval
        "t0_us": t0_us,
        "clip_id": f"custom_segment_frame_{frame_idx}"
    }

# ---
# How to use
# ---
if __name__ == "__main__":
    # Example usage:
    segment_dir = "../../datasets/route_1/segment_00" # path relative to notebook
    frame_idx = 100 
    
    # data = load_custom_dataset(segment_dir, frame_idx)
    #
    # messages = helper.create_message(data["image_frames"].flatten(0, 1))
    # ... rest of the code ...

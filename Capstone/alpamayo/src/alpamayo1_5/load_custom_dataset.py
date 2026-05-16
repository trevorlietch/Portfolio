# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom dataset loader for Alpamayo inference on route folders.

import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps


CAMERA_DIRS = [
    ("raw_left", 0),   # cross-left
    ("raw_front", 1),  # swapped: front-tele is now main
    ("raw_right", 2),  # cross-right
    ("raw", 6),        # swapped: front-wide is now tele
]

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(MODULE_DIR)
ALPAMAYO_DIR = os.path.dirname(SRC_DIR)
PROJECT_DIR = os.path.dirname(ALPAMAYO_DIR)
NOTEBOOKS_DIR = os.path.join(ALPAMAYO_DIR, "notebooks")

ROUTE_CONTEXT_CACHE: dict[str, dict] = {}


def _timestamp_seconds(data: dict) -> float:
    """Return telemetry timestamp in seconds."""
    if "timestamp_seconds" in data:
        return float(data["timestamp_seconds"])
    if "timestamp_eof" in data:
        return float(data["timestamp_eof"]) * 1e-9
    raise KeyError("Telemetry entry is missing timestamp_seconds/timestamp_eof")


def _yaw_rate_rad_s(data: dict, speed_m_s: float) -> float:
    """Use measured yaw rate when available, otherwise bicycle-model fallback."""
    yaw_rate = float(data.get("yaw_rate", 0.0))
    steer_deg = float(data.get("steering_angle_deg", 0.0))
    if abs(yaw_rate) < 1e-4 and abs(steer_deg) > 0.5:
        steer_rad = np.deg2rad(steer_deg) / 15.49
        yaw_rate = speed_m_s * np.tan(steer_rad) / 2.7
    return yaw_rate


def _load_telemetry_series(telemetry_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load timestamps, longitudinal speed, and yaw rate from telemetry."""
    json_files = sorted(
        name for name in os.listdir(telemetry_dir) if name.lower().endswith(".json")
    )
    if not json_files:
        raise FileNotFoundError(f"No telemetry json files found in {telemetry_dir}")

    timestamps_s = []
    speeds_m_s = []
    yaw_rates_rad_s = []

    for name in json_files:
        path = os.path.join(telemetry_dir, name)
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        speed_m_s = float(data.get("v_ego", 0.0))
        yaw_rate_rad_s = _yaw_rate_rad_s(data, speed_m_s)

        if data.get("gear_shifter") == "reverse":
            speed_m_s = -speed_m_s
            yaw_rate_rad_s = -yaw_rate_rad_s

        timestamps_s.append(_timestamp_seconds(data))
        speeds_m_s.append(speed_m_s)
        yaw_rates_rad_s.append(yaw_rate_rad_s)

    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    speeds_m_s = np.asarray(speeds_m_s, dtype=np.float64)
    yaw_rates_rad_s = np.asarray(yaw_rates_rad_s, dtype=np.float64)

    if np.any(np.diff(timestamps_s) < 0):
        raise RuntimeError("Telemetry timestamps are not monotonic")

    return timestamps_s, speeds_m_s, yaw_rates_rad_s


def _resolve_segment_dir(segment_dir: str) -> str:
    """Resolve a segment path against common repo roots and notebook locations."""
    if os.path.isabs(segment_dir):
        return segment_dir

    candidates = []
    seen = set()

    def add_candidate(path: str) -> None:
        norm = os.path.normpath(os.path.abspath(path))
        if norm not in seen:
            seen.add(norm)
            candidates.append(norm)

    def resolve_with_trimmed_components(path: str) -> str | None:
        """Resolve an absolute path when one or more real components have trailing whitespace."""
        normalized = os.path.normpath(os.path.abspath(path))
        drive, tail = os.path.splitdrive(normalized)
        root = drive + os.sep if drive else os.sep
        rel_tail = tail[len(os.sep):] if tail.startswith(os.sep) else tail
        parts = [part for part in rel_tail.split(os.sep) if part]

        current = root
        for part in parts:
            exact = os.path.join(current, part)
            if os.path.exists(exact):
                current = exact
                continue

            try:
                entries = os.listdir(current)
            except OSError:
                return None

            matches = [name for name in entries if name.rstrip() == part.rstrip()]
            if not matches:
                return None

            matches.sort()
            current = os.path.join(current, matches[0])

        return current

    add_candidate(segment_dir)
    add_candidate(os.path.join(PROJECT_DIR, segment_dir))
    add_candidate(os.path.join(ALPAMAYO_DIR, segment_dir))
    add_candidate(os.path.join(NOTEBOOKS_DIR, segment_dir))

    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "telemetry")):
            return candidate
        trimmed_candidate = resolve_with_trimmed_components(candidate)
        if trimmed_candidate and os.path.isdir(os.path.join(trimmed_candidate, "telemetry")):
            return trimmed_candidate

    return candidates[0]


def _discover_route_segments(segment_dir: str) -> tuple[list[str], str]:
    """Return the ordered segment directories for a route and a stable cache key."""
    abs_segment_dir = os.path.abspath(segment_dir)
    segment_name = os.path.basename(abs_segment_dir)
    parent_dir = os.path.dirname(abs_segment_dir)

    if not segment_name.startswith("segment_"):
        return [abs_segment_dir], abs_segment_dir

    segment_dirs = [
        os.path.join(parent_dir, name)
        for name in sorted(os.listdir(parent_dir))
        if name.startswith("segment_") and os.path.isdir(os.path.join(parent_dir, name))
    ]
    segment_dirs = [os.path.abspath(path) for path in segment_dirs]

    if abs_segment_dir not in segment_dirs:
        return [abs_segment_dir], abs_segment_dir

    return segment_dirs, parent_dir


def _load_route_context(segment_dir: str) -> dict:
    """Load and cache route-wide telemetry so GT can cross segment boundaries."""
    segment_dirs, cache_key = _discover_route_segments(segment_dir)
    cached = ROUTE_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    timestamps_parts = []
    speeds_parts = []
    yaw_rates_parts = []
    frame_records: list[tuple[str, int]] = []
    segment_start_indices: dict[str, int] = {}

    for seg_dir in segment_dirs:
        telemetry_dir = os.path.join(seg_dir, "telemetry")
        if not os.path.isdir(telemetry_dir):
            continue

        timestamps_s, speeds_m_s, yaw_rates_rad_s = _load_telemetry_series(telemetry_dir)
        if timestamps_s.size == 0:
            continue

        segment_start_indices[seg_dir] = len(frame_records)
        frame_records.extend((seg_dir, local_idx) for local_idx in range(len(timestamps_s)))
        timestamps_parts.append(timestamps_s)
        speeds_parts.append(speeds_m_s)
        yaw_rates_parts.append(yaw_rates_rad_s)

    if not timestamps_parts:
        raise FileNotFoundError(f"No telemetry json files found for route context rooted at {segment_dir}")

    timestamps_s = np.concatenate(timestamps_parts)
    speeds_m_s = np.concatenate(speeds_parts)
    yaw_rates_rad_s = np.concatenate(yaw_rates_parts)

    if np.any(np.diff(timestamps_s) < 0):
        raise RuntimeError("Route telemetry timestamps are not monotonic")

    absolute_xyz, absolute_theta = _integrate_segment_pose(
        timestamps_s,
        speeds_m_s,
        yaw_rates_rad_s,
    )

    context = {
        "timestamps_s": timestamps_s,
        "absolute_xyz": absolute_xyz,
        "absolute_theta": absolute_theta,
        "frame_records": frame_records,
        "segment_start_indices": segment_start_indices,
    }
    ROUTE_CONTEXT_CACHE[cache_key] = context
    return context


def _integrate_segment_pose(
    timestamps_s: np.ndarray,
    speeds_m_s: np.ndarray,
    yaw_rates_rad_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate absolute XY pose over the segment."""
    n = timestamps_s.shape[0]
    xyz = np.zeros((n, 3), dtype=np.float64)
    theta = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        dt = max(float(timestamps_s[i] - timestamps_s[i - 1]), 0.0)
        v_mid = 0.5 * (speeds_m_s[i - 1] + speeds_m_s[i])
        w_mid = 0.5 * (yaw_rates_rad_s[i - 1] + yaw_rates_rad_s[i])
        theta_mid = theta[i - 1] + 0.5 * w_mid * dt

        xyz[i, 0] = xyz[i - 1, 0] + v_mid * np.cos(theta_mid) * dt
        xyz[i, 1] = xyz[i - 1, 1] + v_mid * np.sin(theta_mid) * dt
        theta[i] = theta[i - 1] + w_mid * dt

    return xyz, np.unwrap(theta)


def _nearest_indices(timestamps_s: np.ndarray, query_times_s: np.ndarray) -> np.ndarray:
    """Find nearest frame indices for a set of timestamps."""
    indices = np.searchsorted(timestamps_s, query_times_s, side="left")
    indices = np.clip(indices, 0, len(timestamps_s) - 1)
    prev_indices = np.clip(indices - 1, 0, len(timestamps_s) - 1)

    choose_prev = np.abs(query_times_s - timestamps_s[prev_indices]) <= np.abs(
        timestamps_s[indices] - query_times_s
    )
    return np.where(choose_prev, prev_indices, indices)


def _sample_local_trajectory(
    timestamps_s: np.ndarray,
    absolute_xyz: np.ndarray,
    absolute_theta: np.ndarray,
    query_times_s: np.ndarray,
    t0_time_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the integrated pose at query times and transform into the t0 ego frame."""
    query_times_s = np.clip(query_times_s, timestamps_s[0], timestamps_s[-1])

    x = np.interp(query_times_s, timestamps_s, absolute_xyz[:, 0])
    y = np.interp(query_times_s, timestamps_s, absolute_xyz[:, 1])
    theta = np.interp(query_times_s, timestamps_s, absolute_theta)

    t0_x = float(np.interp(t0_time_s, timestamps_s, absolute_xyz[:, 0]))
    t0_y = float(np.interp(t0_time_s, timestamps_s, absolute_xyz[:, 1]))
    t0_theta = float(np.interp(t0_time_s, timestamps_s, absolute_theta))

    dx = x - t0_x
    dy = y - t0_y

    c = np.cos(t0_theta)
    s = np.sin(t0_theta)
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    theta_local = theta - t0_theta

    xyz_local = np.stack(
        [x_local, y_local, np.zeros_like(x_local, dtype=np.float64)],
        axis=-1,
    )
    return xyz_local, theta_local


def _theta_to_rotations(theta_local: np.ndarray) -> np.ndarray:
    """Convert yaw angles into 3x3 rotation matrices."""
    c = np.cos(theta_local)
    s = np.sin(theta_local)
    rot = np.zeros((theta_local.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = c
    rot[:, 0, 1] = -s
    rot[:, 1, 0] = s
    rot[:, 1, 1] = c
    rot[:, 2, 2] = 1.0
    return rot


def load_custom_dataset(
    segment_dir: str,
    frame_idx: int,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    frame_stride: int = 1,
    visual_stride: int = 1,
    exclude_cameras: list[int] = None,
):
    """
    Load a route segment and resample kinematics to Alpamayo's expected 10Hz layout.

    The synced route folders are not guaranteed to be 20Hz openpilot frame sequences,
    so we use route telemetry timestamps to reconstruct a continuous local trajectory
    across neighboring segment_* folders when available, then sample:
      - history at [..., t0-0.2, t0-0.1, t0]
      - future at [t0+0.1, ..., t0+6.4]

    ``frame_stride`` and ``visual_stride`` are legacy compatibility arguments and
    are ignored by this loader.
    """
    del frame_stride, visual_stride  # Kept in the signature for notebook compatibility.

    resolved_segment_dir = _resolve_segment_dir(segment_dir)
    abs_segment_dir = os.path.abspath(resolved_segment_dir)
    telemetry_dir = os.path.join(abs_segment_dir, "telemetry")

    if not os.path.isdir(telemetry_dir):
        raise FileNotFoundError(f"Telemetry directory not found: {telemetry_dir}")

    route_context = _load_route_context(abs_segment_dir)
    segment_start_indices = route_context["segment_start_indices"]
    if abs_segment_dir not in segment_start_indices:
        raise RuntimeError(f"Segment {segment_dir} is missing from the route telemetry context")

    local_timestamps_s, _, _ = _load_telemetry_series(telemetry_dir)
    if frame_idx < 0 or frame_idx >= len(local_timestamps_s):
        raise IndexError(
            f"frame_idx {frame_idx} is out of range for {segment_dir} "
            f"(0..{len(local_timestamps_s) - 1})"
        )

    timestamps_s = route_context["timestamps_s"]
    absolute_xyz = route_context["absolute_xyz"]
    absolute_theta = route_context["absolute_theta"]
    frame_records = route_context["frame_records"]
    global_frame_idx = segment_start_indices[abs_segment_dir] + frame_idx

    t0_s = float(timestamps_s[global_frame_idx])
    history_offsets_s = np.arange(
        -(num_history_steps - 1) * time_step,
        0.5 * time_step,
        time_step,
        dtype=np.float64,
    )
    future_offsets_s = np.arange(
        time_step,
        (num_future_steps + 0.5) * time_step,
        time_step,
        dtype=np.float64,
    )

    history_times_s = t0_s + history_offsets_s
    future_times_s = t0_s + future_offsets_s

    ego_history_xyz_np, ego_history_theta_np = _sample_local_trajectory(
        timestamps_s=timestamps_s,
        absolute_xyz=absolute_xyz,
        absolute_theta=absolute_theta,
        query_times_s=history_times_s,
        t0_time_s=t0_s,
    )
    ego_future_xyz_np, ego_future_theta_np = _sample_local_trajectory(
        timestamps_s=timestamps_s,
        absolute_xyz=absolute_xyz,
        absolute_theta=absolute_theta,
        query_times_s=future_times_s,
        t0_time_s=t0_s,
    )

    ego_history_rot_np = _theta_to_rotations(ego_history_theta_np)
    ego_future_rot_np = _theta_to_rotations(ego_future_theta_np)

    ego_history_xyz = torch.from_numpy(ego_history_xyz_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    ego_history_rot = torch.from_numpy(ego_history_rot_np).unsqueeze(0).unsqueeze(0)
    ego_future_xyz = torch.from_numpy(ego_future_xyz_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    ego_future_rot = torch.from_numpy(ego_future_rot_np).unsqueeze(0).unsqueeze(0)

    # Load images closest to the official Alpamayo image times [t0-0.3, t0-0.2, t0-0.1, t0].
    num_visual_frames = 4
    image_offsets_s = np.arange(
        -(num_visual_frames - 1) * time_step,
        0.5 * time_step,
        time_step,
        dtype=np.float64,
    )
    image_times_s = t0_s + image_offsets_s
    image_indices = _nearest_indices(timestamps_s, image_times_s)

    all_camera_frames = []
    all_camera_indices = []

    if exclude_cameras is None:
        exclude_cameras = []

    for dir_name, cam_idx in CAMERA_DIRS:
        if cam_idx in exclude_cameras:
            continue
        cam_dir = os.path.join(abs_segment_dir, dir_name)
        if not os.path.isdir(cam_dir):
            continue
        if not any(name.lower().endswith(".png") for name in os.listdir(cam_dir)):
            continue

        images = []
        for idx in image_indices:
            image_segment_dir, image_local_idx = frame_records[int(idx)]
            img_path = os.path.join(image_segment_dir, dir_name, f"{image_local_idx:06d}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = ImageOps.pad(img, (640, 480), method=Image.Resampling.BILINEAR)
                img_np = np.array(img)
            else:
                img_np = np.zeros((480, 640, 3), dtype=np.uint8)
            images.append(img_np)

        cam_tensor = torch.tensor(np.stack(images), dtype=torch.uint8).permute(0, 3, 1, 2)
        all_camera_frames.append(cam_tensor)
        all_camera_indices.append(cam_idx)

    if not all_camera_frames:
        raise FileNotFoundError(
            f"No camera directories with images found in {segment_dir}. Expected at least raw/."
        )

    image_frames = torch.stack(all_camera_frames, dim=0)
    camera_indices = torch.tensor(all_camera_indices, dtype=torch.int64)
    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]

    chosen_image_times_us = np.round(timestamps_s[image_indices] * 1_000_000.0).astype(np.int64)
    absolute_timestamps = torch.tensor(
        np.tile(chosen_image_times_us, (len(all_camera_frames), 1)),
        dtype=torch.int64,
    )[sort_order]
    relative_timestamps = (absolute_timestamps - absolute_timestamps.min()).float() * 1e-6

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "ego_future_xyz": ego_future_xyz,
        "ego_future_rot": ego_future_rot,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": absolute_timestamps,
        "t0_us": int(round(t0_s * 1_000_000.0)),
        "clip_id": f"custom_{os.path.basename(abs_segment_dir)}_frame_{frame_idx}",
    }


if __name__ == "__main__":
    segment_dir = "../../datasets/route_1/segment_00"
    frame_idx = 100
    data = load_custom_dataset(segment_dir, frame_idx)
    print(data["ego_history_xyz"].shape, data["ego_future_xyz"].shape)

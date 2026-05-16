"""Navigation command inference from ground-truth future trajectories."""

from __future__ import annotations

import numpy as np


def infer_navigation_command(
    gt_xyz: np.ndarray,
    lookahead_frames: int = 200,
    turn_angle_threshold_deg: float = 15.0,
    turn_start_angle_threshold_deg: float = 2.0,
    min_detection_distance_m: float = 5.0,
    min_turn_start_distance_m: float = 1.0,
    max_command_distance_m: float = 80.0,
    immediate_turn_distance_m: float = 3.0,
) -> str:
    """Return a stateless nav command for the current future path.

    The command is intentionally derived only from the current frame's ground
    truth trajectory. Batch export advances frame-by-frame, so carrying a turn
    command after the trajectory has become straight makes Alpamayo keep turning
    after the car has exited the intersection.
    """
    if gt_xyz is None or gt_xyz.shape[0] == 0:
        return "Go Straight"

    check_frames = min(lookahead_frames, gt_xyz.shape[0])
    xs = gt_xyz[:check_frames, 0]
    ys = gt_xyz[:check_frames, 1]

    path_angles = np.degrees(np.arctan2(ys, xs))
    distances = np.hypot(xs, ys)

    far_idx = distances > min_detection_distance_m
    if not np.any(far_idx):
        return "Go Straight"

    if np.max(path_angles[far_idx]) > turn_angle_threshold_deg:
        raw_nav_cmd = "Turn left"
        turn_sign = 1
    elif np.min(path_angles[far_idx]) < -turn_angle_threshold_deg:
        raw_nav_cmd = "Turn right"
        turn_sign = -1
    else:
        return "Go Straight"

    turn_start_idx = None
    for i in range(check_frames):
        if distances[i] < min_turn_start_distance_m:
            continue
        if turn_sign * path_angles[i] > turn_start_angle_threshold_deg:
            turn_start_idx = i
            break

    if turn_start_idx is None:
        return "Go Straight"

    turn_dist_m = max(0.0, float(xs[turn_start_idx]))
    if turn_dist_m >= max_command_distance_m:
        return "Go Straight"
    if turn_dist_m <= immediate_turn_distance_m:
        return raw_nav_cmd
    return f"{raw_nav_cmd} in {int(turn_dist_m)}m"

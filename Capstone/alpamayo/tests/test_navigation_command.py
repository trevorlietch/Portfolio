import os
import sys

import numpy as np


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from alpamayo1_5.navigation_command import infer_navigation_command


def _trajectory(xs, ys):
    return np.column_stack([xs, ys, np.zeros_like(xs)])


def test_straight_trajectory_returns_go_straight():
    xs = np.linspace(1.0, 80.0, 64)
    gt_xyz = _trajectory(xs, np.zeros_like(xs))

    assert infer_navigation_command(gt_xyz) == "Go Straight"


def test_upcoming_left_turn_returns_turn_command_with_distance():
    xs = np.linspace(1.0, 80.0, 64)
    ys = np.maximum(0.0, xs - 20.0) * 0.45
    gt_xyz = _trajectory(xs, ys)

    assert infer_navigation_command(gt_xyz).startswith("Turn left in ")


def test_upcoming_right_turn_returns_turn_command_with_distance():
    xs = np.linspace(1.0, 80.0, 64)
    ys = -np.maximum(0.0, xs - 20.0) * 0.45
    gt_xyz = _trajectory(xs, ys)

    assert infer_navigation_command(gt_xyz).startswith("Turn right in ")


def test_post_intersection_straight_trajectory_does_not_hold_previous_turn():
    xs = np.linspace(1.0, 80.0, 64)
    left_turn = _trajectory(xs, np.maximum(0.0, xs - 20.0) * 0.45)
    post_turn_straight = _trajectory(xs, np.zeros_like(xs))

    assert infer_navigation_command(left_turn).startswith("Turn left")
    assert infer_navigation_command(post_turn_straight) == "Go Straight"

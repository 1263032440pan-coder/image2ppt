"""Per-component shape classifiers used by the inventory builder.

These describe how a detected crop looks (card / panel or connector).
They do not decide whether a large connected component is atomic; the
inventory builder splits large regions first, then uses the role to choose
asset routing and whether a composite backplate is appropriate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PAGE_DIR = Path(__file__).resolve().parents[1]
if str(PAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PAGE_DIR))

from _heuristics import s_area, s_length  # noqa: E402


def is_connector_like(crop: np.ndarray, scale: float) -> bool:
    if crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if max(w, h) < s_length(26, scale):
        return False
    if w * h > s_area(24000, scale):
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    diff_white = np.abs(crop.astype(np.int16) - 255).max(axis=2)
    fg = (
        ((gray < 235) & (diff_white > 8))
        | ((hsv[:, :, 1] > 30) & (diff_white > 6))
    )
    fg_count = int(fg.sum())
    if fg_count < s_area(12, scale):
        return False
    density = fg_count / float(max(1, w * h))
    aspect = max(w, h) / float(max(1, min(w, h)))
    if aspect >= 5.0 and density <= 0.60:
        return True
    if density > 0.28:
        return False
    ys, xs = np.where(fg)
    if len(xs) < 8:
        return False
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    pts -= pts.mean(axis=0, keepdims=True)
    cov = (pts.T @ pts) / max(1, len(pts) - 1)
    vals = np.linalg.eigvalsh(cov)
    if vals[0] <= 1e-3:
        return True
    return bool(vals[1] / vals[0] >= 9.0)


def is_container_like(crop: np.ndarray, scale: float) -> bool:
    if crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if w < s_length(42, scale) or h < s_length(22, scale):
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    diff_white = np.abs(crop.astype(np.int16) - 255).max(axis=2)
    fg = (
        ((gray < 248) & (diff_white > 8))
        | ((hsv[:, :, 1] > 10) & (diff_white > 6))
    )
    band = max(2, min(max(2, min(w, h) // 10), s_length(10, scale)))
    top = float(np.any(fg[:band, :], axis=0).sum()) / max(1, w)
    bottom = float(np.any(fg[-band:, :], axis=0).sum()) / max(1, w)
    left = float(np.any(fg[:, :band], axis=1).sum()) / max(1, h)
    right = float(np.any(fg[:, -band:], axis=1).sum()) / max(1, h)
    edge_like = (
        sum(v >= 0.24 for v in (top, bottom, left, right)) >= 3
        and max(top, bottom) >= 0.45
        and max(left, right) >= 0.45
    )
    fill_density = float(fg.sum()) / float(max(1, w * h))
    fg_hsv = hsv[fg]
    dominant_fill_fraction = 0.0
    if len(fg_hsv):
        # A dense photograph or detailed screenshot is not a card merely
        # because most pixels differ from white. A real filled panel usually
        # has one dominant fill colour with sparse text and icon colours.
        h_bin = (fg_hsv[:, 0] // 12).astype(np.int32)
        s_bin = (fg_hsv[:, 1] // 32).astype(np.int32)
        v_bin = (fg_hsv[:, 2] // 32).astype(np.int32)
        color_bins = h_bin * 64 + s_bin * 8 + v_bin
        color_counts = np.bincount(color_bins, minlength=15 * 64)
        dominant_fill_fraction = float(color_counts.max()) / len(fg_hsv)
    filled_shape = (
        fill_density >= 0.45
        and dominant_fill_fraction >= 0.58
        and max(w, h) >= s_length(70, scale)
        and max(w, h) / float(max(1, min(w, h))) <= 8.0
    )
    return bool(edge_like or filled_shape)


def foreground_role_for_box(img: np.ndarray, x1: int, y1: int,
                            x2: int, y2: int, scale: float) -> str | None:
    crop = img[y1:y2, x1:x2]
    if is_container_like(crop, scale):
        return "container"
    if is_connector_like(crop, scale):
        return "connector"
    return None

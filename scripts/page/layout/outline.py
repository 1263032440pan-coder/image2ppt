"""Outline / card shape helpers.

When a detected component looks like a card border (outline mask
exists), these helpers decide whether to:
  * emit it as a native PPT rounded-rectangle shape, or
  * keep the original alpha-masked image crop, or
  * split a tall outline into multiple stacked sub-cards.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from shared.geometry import bgr_to_hex as _bgr_to_hex


def _sample_outline_color(
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
    mask_path: str | None,
) -> str:
    """Sample a card/frame line colour from its alpha mask when present."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h_img, w_img = source.shape[:2]
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return "#C8D7EA"
    crop = source[y1:y2, x1:x2]
    pixels = None
    if mask_path and Path(mask_path).exists():
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None and m.shape[:2] == source.shape[:2]:
            opaque = m[y1:y2, x1:x2] > 16
            if int(opaque.sum()) >= 8:
                pixels = crop[opaque]
    if pixels is None or len(pixels) == 0:
        band = max(2, min(6, min(y2 - y1, x2 - x1) // 18))
        edge = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        edge[:band, :] = True
        edge[-band:, :] = True
        edge[:, :band] = True
        edge[:, -band:] = True
        pixels = crop[edge]
    if len(pixels) == 0:
        return "#C8D7EA"
    med = np.median(pixels.reshape(-1, 3), axis=0)
    # If the mask sampled mostly white interior pixels, fall back to
    # the deck's common pale blue frame colour instead of emitting
    # invisible white lines.
    if int(np.max(np.abs(med.astype(int) - 255))) <= 8:
        return "#C8D7EA"
    return _bgr_to_hex(med.astype(int))


def _sample_card_fill_color(
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> str:
    """Sample a pale card/panel fill colour from inside an outline bbox."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h_img, w_img = source.shape[:2]
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return "#FFFFFF"
    w = x2 - x1
    h = y2 - y1
    pad = max(6, min(18, min(w, h) // 10))
    inner = source[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
    if inner.size == 0:
        inner = source[y1:y2, x1:x2]
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    light_bg = (gray > 220) & (hsv[:, :, 1] < 70)
    pixels = inner[light_bg]
    if len(pixels) < max(20, int(0.05 * inner.shape[0] * inner.shape[1])):
        pixels = inner.reshape(-1, 3)
    med = np.median(pixels.reshape(-1, 3), axis=0).astype(int)
    return _bgr_to_hex(med)


def _simple_internal_shape(
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> dict | None:
    """Describe a flat internal badge/panel as a native PPT shape.

    Internal detections can also contain multicolor illustrations. Convert
    only a dominant, nearly uniform fill and a clear ellipse/rectangle;
    keep everything else as an image asset.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    crop = source[max(0, y1):min(source.shape[0], y2),
                  max(0, x1):min(source.shape[1], x2)]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    h, w = crop.shape[:2]
    cy1, cy2 = int(0.12 * h), max(int(0.88 * h), int(0.12 * h) + 1)
    cx1, cx2 = int(0.12 * w), max(int(0.88 * w), int(0.12 * w) + 1)
    sample = crop[cy1:cy2, cx1:cx2]
    quant = (sample.reshape(-1, 3).astype(np.uint16) // 16).astype(np.uint8)
    colors, counts = np.unique(quant, axis=0, return_counts=True)
    color_bin = colors[int(np.argmax(counts))].astype(np.int16)
    sample_flat = sample.reshape(-1, 3).astype(np.int16)
    sample_delta = np.abs(sample_flat - (color_bin * 16 + 8)).max(axis=1)
    support = float((sample_delta <= 28).sum()) / max(1, len(sample_delta))
    # If icons, labels or markings sit on a filled carrier, leave it as an
    # image asset so those details survive. Native conversion is reserved
    # for virtually flat fills.
    if support < 0.92:
        return None

    fill_bgr = np.median(sample_flat[sample_delta <= 28], axis=0).astype(np.uint8)
    delta = np.abs(crop.astype(np.int16) - fill_bgr.astype(np.int16)).max(axis=2)
    mask = (delta <= 32).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(contour)
    if bw < 0.72 * w or bh < 0.72 * h:
        return None
    area_ratio = cv2.contourArea(contour) / float(max(1, bw * bh))
    if area_ratio < 0.62:
        return None
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, max(1.0, 0.025 * perimeter), True)
    aspect = bw / float(max(1, bh))
    shape = None
    if (0.82 <= aspect <= 1.22 and 0.68 <= area_ratio <= 0.88
            and len(approx) >= 6):
        shape = "oval"
    elif area_ratio >= 0.84 and len(approx) <= 6:
        shape = "rect"
    if shape is None:
        return None
    return {
        "shape": shape,
        "fill": _bgr_to_hex(fill_bgr.astype(int)),
        "line": None,
    }


def _outline_should_be_native_shape(
        bbox: tuple[int, int, int, int]) -> bool:
    """Use native round-rect only for card-like outlines.

    Near-square outline masks are often circular icon containers or
    central rings. Rendering those as a PowerPoint rounded rectangle
    creates an extra visible box, so they should stay as alpha-masked
    PNGs that preserve the original contour.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect = w / float(h)
    return aspect >= 1.45 or aspect <= 0.69


def _outline_should_keep_full_crop(
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> bool:
    """Keep a filled card as an image crop when native shapes are disabled."""
    fill = _sample_card_fill_color(source, bbox).lstrip("#")
    if len(fill) != 6:
        return False
    rgb = np.array([int(fill[i:i + 2], 16) for i in (0, 2, 4)], dtype=int)
    return int(np.max(np.abs(rgb - 255))) > 7


def _split_filled_outline_rows(
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    """Split stacked filled sub-cards inside one detected outline bbox."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    crop = source[y1:y2, x1:x2]
    if crop.size == 0:
        return [(x1, y1, x2, y2)]
    h, w = crop.shape[:2]
    if w < 220 or h < 90:
        return [(x1, y1, x2, y2)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    diff_white = np.abs(crop.astype(int) - 255).max(axis=2)
    panel = (
        ((gray < 253) & (diff_white > 4))
        | ((hsv[:, :, 1] > 6) & (diff_white > 3))
    )
    row_count = panel.sum(axis=1)
    active = row_count > max(18, int(0.08 * w))

    ranges: list[tuple[int, int]] = []
    start: int | None = None
    gap_start: int | None = None
    min_gap = 4
    for idx, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = idx
            gap_start = None
        elif start is not None and gap_start is None:
            gap_start = idx
        elif (start is not None and gap_start is not None
              and idx - gap_start >= min_gap):
            if gap_start - start >= 28:
                ranges.append((start, gap_start))
            start = None
            gap_start = None
    if start is not None:
        end = (gap_start if gap_start is not None
               and h - gap_start >= min_gap else h)
        if end - start >= 28:
            ranges.append((start, end))

    if len(ranges) <= 1:
        return [(x1, y1, x2, y2)]
    # Keep only real card-height rows; tiny rule/text fragments are not
    # independent panels.
    boxes = [(x1, y1 + rs, x2, y1 + re)
             for rs, re in ranges if re - rs >= 36]
    if len(boxes) <= 1:
        return [(x1, y1, x2, y2)]
    return boxes

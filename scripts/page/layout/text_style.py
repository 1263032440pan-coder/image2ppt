"""Line-level and mixed-run text style detection.

Samples the BACKGROUND from a ring outside the OCR bbox first; pixels
inside the bbox far from the ring colour are text strokes. Their median
gives the text colour. Font family, weight, and italic style are selected
by local font rendering fit in the downstream layout stage. When `text`
and character boxes are supplied, it samples per-character color so
colored text can be emitted as editable PowerPoint runs.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from layout.color import (
    _classify_text_color,
    _normalize_run_colors,
)
from layout.text_runs import _per_char_runs


def detect_text_style(bbox: list[int], orig_img: np.ndarray,
                      text: str | None = None,
                      char_boxes: list[list[int]] | None = None,
                      words: list[str] | None = None,
                      word_boxes: list[list[int]] | None = None) -> dict:
    """Sample text color from the image; defer font weight to render fitting.

    Sample the BACKGROUND from a 4 px ring just outside the bbox first.
    Pixels INSIDE the bbox close to the bg color (within ±35 per channel)
    are considered background; the rest are text strokes. Median of the
    text-stroke pixels gives the text color. The legacy `bold` result is
    kept as False for layout-schema compatibility; it is not image-inferred.

    When `text` is provided, also runs per-character color sampling and,
    if multiple distinct color runs are detected in the bbox, returns
    them under the `runs` key.
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = orig_img.shape[:2]
    region = orig_img[y1:y2, x1:x2]
    if region.size == 0:
        return {"color": "#054798", "bold": False}

    # 6-px ring offset 2 px from the bbox. The offset keeps the inner
    # row of the ring outside the antialiased character edge, which
    # would otherwise pollute the bg median with text-colour pixels.
    gap = 2
    ring = 6
    inner_top = max(0, y1 - gap)
    inner_bot = min(h_img, y2 + gap)
    inner_left = max(0, x1 - gap)
    inner_right = min(w_img, x2 + gap)
    bg_samples = []
    if inner_top - ring >= 0:
        bg_samples.append(
            orig_img[inner_top - ring:inner_top, inner_left:inner_right])
    if inner_bot + ring <= h_img:
        bg_samples.append(
            orig_img[inner_bot:inner_bot + ring, inner_left:inner_right])
    if inner_left - ring >= 0:
        bg_samples.append(
            orig_img[inner_top:inner_bot, inner_left - ring:inner_left])
    if inner_right + ring <= w_img:
        bg_samples.append(
            orig_img[inner_top:inner_bot, inner_right:inner_right + ring])
    if bg_samples:
        bg_pixels = np.concatenate(
            [s.reshape(-1, 3) for s in bg_samples if s.size > 0])
        # Vote majority bg via quantised mode, keep only pixels close
        # to mode, then take their median. Excludes text-colour leaks
        # from antialiased character strokes bleeding into the ring.
        quant = (bg_pixels // 16) * 16
        mode_q = np.array(Counter(map(tuple, quant)).most_common(1)[0][0])
        diff = np.abs(bg_pixels.astype(int) - mode_q).max(axis=1)
        close = bg_pixels[diff <= 30]
        if len(close) >= max(8, int(len(bg_pixels) * 0.2)):
            bg = np.median(close, axis=0)
        else:
            bg = np.median(bg_pixels, axis=0)
    else:
        bg = np.array([255.0, 255.0, 255.0])

    # Tight-coloured-container case: when an OCR bbox almost exactly
    # fills a coloured badge (e.g. white `Rhombus` text on a dark-blue
    # rounded rect), the ring sample lands OUTSIDE the badge and reports
    # white. Detect via inside-bbox saturation majority.
    inner_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat_mask = inner_hsv[:, :, 1] > 50
    total_px = region.shape[0] * region.shape[1]
    # Threshold tuned for: saturated TEXT on neutral bg ~20-30 %;
    # saturated CONTAINER fill with light text ~60-80 %. 0.55 sits
    # between the two.
    is_container_like = int(sat_mask.sum()) > 0.55 * total_px
    if is_container_like:
        # Disambiguate dense CJK text from real containers: dense
        # strokes are many small blobs, real container is one big blob.
        n_comp, _, stats, _ = cv2.connectedComponentsWithStats(
            sat_mask.astype(np.uint8) * 255, 8)
        if n_comp > 1:
            largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
            total_sat = int(sat_mask.sum())
            if largest_area < 0.55 * total_sat:
                is_container_like = False
    if is_container_like:
        container_bg = np.median(region[sat_mask], axis=0)
        if int(np.max(np.abs(
                container_bg.astype(int) - bg.astype(int)))) > 40:
            bg = container_bg

    # Pixels within bbox far from bg are text strokes.
    diff = np.abs(region.astype(int) - bg).max(axis=2)
    text_mask = diff > 35
    if int(text_mask.sum()) < 5:
        # Fallback: take darkest 25 % vs bg.
        thr = max(15, int(np.percentile(diff, 75)))
        text_mask = diff > thr
        if int(text_mask.sum()) < 5:
            return {"color": "#054798", "bold": False}

    text_pixels = region[text_mask]
    m = np.median(text_pixels, axis=0).astype(int)
    b, g, r = int(m[0]), int(m[1]), int(m[2])
    color = _classify_text_color(r, g, b, bg)

    # ink_h: tight vertical extent of strokes inside the bbox. Used as
    # a decoration-padding probe by the caller. It does not decide font
    # weight; bold/regular is chosen by local render fitting downstream.
    ink_h: int | None = None
    row_ink = text_mask.sum(axis=1) >= 2
    rows = np.where(row_ink)[0]
    if rows.size >= 2:
        ink_h = int(rows.max() - rows.min() + 1)

    result = {"color": color, "bold": False, "ink_h": ink_h}

    if text and len(text) >= 2:
        runs = _per_char_runs(bbox, text, orig_img, bg, color,
                              char_boxes=char_boxes,
                              words=words, word_boxes=word_boxes)
        _normalize_run_colors(runs, color, bg)
        if "".join(str(run.get("text", "")) for run in runs) == text:
            if len(runs) > 1:
                result["runs"] = runs
    return result


def _glyph_height_in_bbox(orig_img: np.ndarray, bg: np.ndarray,
                          bbox: tuple[int, int, int, int]) -> int | None:
    """Measure vertical ink extent inside a text bbox.

    PaddleOCR's detection bbox often pads vertically for decorated stat
    numbers — drop shadows, glow effects, gradient borders inflate the
    box past the actual glyph extent. Returns the ink-row count (rows
    containing ≥ 2 stroke px), or None if sampling fails.
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = orig_img.shape[:2]
    x1 = max(0, min(w_img, int(x1))); x2 = max(0, min(w_img, int(x2)))
    y1 = max(0, min(h_img, int(y1))); y2 = max(0, min(h_img, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    region = orig_img[y1:y2, x1:x2]
    if region.size == 0:
        return None
    diff = np.abs(region.astype(int) - bg).max(axis=2)
    mask = diff > 35
    if int(mask.sum()) < 3:
        return None
    row_ink = mask.sum(axis=1) >= 2
    rows = np.where(row_ink)[0]
    if rows.size < 2:
        return None
    return int(rows.max() - rows.min() + 1)

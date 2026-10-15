#!/usr/bin/env python
"""Build a complete element inventory from a cleaned slide image + OCR JSON.

Inventory format (JSON list):
  [
    {"id":"t000","type":"text","text":"...","bbox":[x1,y1,x2,y2],"confidence":1.0},
    {"id":"v000","type":"image","bbox":[x1,y1,x2,y2],"source":"cleaned"},
    ...
  ]

The inventory captures EVERY visible element on the slide. Text comes from
OCR; images come from cv2 connected-component detection on the text-erased
image. The inventory is the single source of truth for downstream PPTX build.

Detection rules:
- Min icon area = 300 px, min dimension = 18 px (allows independent small
  icons but filters noise).
- Morphological close 6x6 BEFORE detection - merges fragments WITHIN one icon
  (e.g. small parts of a complex icon) into a single component, while leaving
  spatially separate independent icons as distinct components.
- Conservative split: each component is checked for white-space gaps >= 10 px
  internally. If found, split along those gaps. If no clear gap, KEEP WHOLE
  (preserves bar charts with arrows, complex compositions).

Usage:
    python build_inventory.py --clean clean.png --ocr ocr.json --out inv.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Re-use erase_text logic for the icon/logo filters, and the icon
# package for sub-shape detection inside parent components.
import sys
_SCRIPTS = Path(__file__).resolve().parent          # .../scripts/pipeline
_SCRIPTS_ROOT = _SCRIPTS.parent                      # .../scripts
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS_ROOT))
from shared.bg_sample import estimate_canvas_bgr as _estimate_background_bgr
from shared.geometry import (
    connector_on_container_border as _connector_on_container_border,
    intersection_area as _intersection_area,
)
from erase_text import (
    detect_logo_strips,
    is_in_logo_zone,
    is_likely_icon,
    preprocess_ocr,
    should_preserve_visual,
)
from _heuristics import pixel_scale, s_area, s_kernel, s_length
from icon import (
    detect_internal_shapes,
    detect_line_art_subicons,
    detect_white_subicons,
    inpaint_region_inplace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build inventory from cleaned image + OCR.")
    parser.add_argument("--clean", required=True, help="Cleaned image path (text erased).")
    parser.add_argument("--source", required=True, help="Original source image (for icon filters).")
    parser.add_argument("--ocr", required=True, help="OCR JSON path.")
    parser.add_argument("--out", required=True, help="Inventory JSON output path.")
    parser.add_argument("--min-area", type=int, default=80,
                        help="Min component area (very loose - only filters noise dots).")
    parser.add_argument("--dilate", type=int, default=6)
    parser.add_argument("--split-gap", type=int, default=12)
    parser.add_argument("--debug-dir",
                        help="Optional debug directory. Writes <stem>_inv.png "
                             "showing detected components, subicons, and OCR "
                             "text overlaid on the source image.")
    parser.add_argument("--masks-dir",
                        help="Save per-component mask PNGs to this "
                             "directory. inventory_to_layout reads them by "
                             "element id (e.g. v005.mask.png) to alpha-key "
                             "the asset crop.")
    return parser.parse_args()


def _foreground_mask(cleaned: np.ndarray, dilate_size: int) -> np.ndarray:
    """Foreground mask used by detect_components and conservative_split.

    Threshold: noticeably different from the corner-estimated canvas
    background plus a Canny edge layer that catches 1-px pale outlines.
    The old white-background test made every pixel of a dark slide look
    like foreground, merging the whole page into one flattened object.
    """
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
    bg = _estimate_background_bgr(cleaned).astype(np.int16)
    diff_bg = np.abs(cleaned.astype(np.int16) - bg).max(axis=2)
    bg_pixel = bg.astype(np.uint8).reshape(1, 1, 3)
    bg_hsv = cv2.cvtColor(bg_pixel, cv2.COLOR_BGR2HSV)[0, 0]
    sat_delta = np.abs(hsv[:, :, 1].astype(np.int16) - int(bg_hsv[1]))
    # Lab distance catches subtle hue changes that are not large in any
    # single BGR channel. Keep the old channel/saturation rules as well,
    # since they are more sensitive to very pale anti-aliased strokes.
    lab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB).astype(np.int32)
    bg_lab = cv2.cvtColor(bg.astype(np.uint8).reshape(1, 1, 3),
                          cv2.COLOR_BGR2LAB)[0, 0].astype(np.int32)
    lab_delta = np.sqrt(np.sum((lab - bg_lab[None, None, :]) ** 2,
                               axis=2))
    global_mask = (
        (diff_bg >= 8)
        | ((diff_bg >= 5) & (sat_delta > 18))
        | (lab_delta >= 7.0)
    )
    # Very pale, low-chroma fills are common slide-wide panel backdrops.
    # Treating every such pixel as foreground connects neighboring cards
    # into one page-sized component. Their boundaries are recovered below
    # by the multi-channel edge pass, while darker/saturated fills remain
    # solid foreground regions.
    pale_backdrop = (gray >= 235) & (hsv[:, :, 1] < 45) & (diff_bg <= 42)
    global_mask &= ~pale_backdrop

    # A local-background residual finds low-contrast cards and decorations
    # on gradients or tinted panels. The canvas-relative mask remains the
    # primary cue so broad background texture does not become foreground.
    scale = max(0.5, cleaned.shape[0] / 720.0)
    local_kernel = max(15, int(round(41 * scale)))
    if local_kernel % 2 == 0:
        local_kernel += 1
    local_bg = cv2.GaussianBlur(cleaned, (local_kernel, local_kernel), 0)
    local_delta = np.abs(cleaned.astype(np.int16)
                         - local_bg.astype(np.int16)).max(axis=2)
    local_mask = (local_delta >= 13) & ~pale_backdrop & (
        (hsv[:, :, 1] > 18) | (gray < 238) | (diff_bg >= 4)
    )
    mask = (global_mask | local_mask).astype(np.uint8) * 255
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90)
    lab8 = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB)
    # Lightly colored boundaries can have almost identical luminance; low
    # threshold Canny passes on Lab chroma preserve those card outlines.
    edges = cv2.bitwise_or(edges, cv2.Canny(lab8[:, :, 1], 10, 32))
    edges = cv2.bitwise_or(edges, cv2.Canny(lab8[:, :, 2], 10, 32))
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.bitwise_or(mask, edges)
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_components(cleaned: np.ndarray, min_area: int, dilate_size: int,
                      foreground_mask: np.ndarray | None = None,
                      *, return_labels: bool = False
                      ) -> list[tuple] | tuple[list[tuple], np.ndarray, list[int]]:
    """Detect ALL connected components. Returns components above noise floor.

    Even a whole-image-sized component is returned (not filtered) so that
    conservative_split can break it apart along white-space gaps. Filtering
    "big" components prematurely loses real content (e.g. when a faint
    decorative element bridges otherwise distinct icons).

    Foreground threshold: see _foreground_mask. Catches:
    - Light fill rects (light blue cards, gray box backgrounds)
    - Subtle line art (white-bg sketches with light strokes)
    - Saturated colored regions (bars, banners, icons)
    """
    mask = (foreground_mask if foreground_mask is not None
            else _foreground_mask(cleaned, dilate_size))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    label_ids = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < min_area:
            continue
        out.append((int(x), int(y), int(x + ww), int(y + hh), int(area)))
        label_ids.append(i)
    if return_labels:
        return out, labels, label_ids
    return out


def _find_gaps(line_has_fg: np.ndarray, min_gap: int) -> list[tuple]:
    """Return list of (start, end) gap intervals in a 1D foreground signal."""
    gaps = []
    in_gap = False
    gs = 0
    n = len(line_has_fg)
    for i in range(n):
        if not line_has_fg[i]:
            if not in_gap:
                gs = i
                in_gap = True
        else:
            if in_gap:
                if i - gs >= min_gap:
                    gaps.append((gs, i))
                in_gap = False
    return gaps


def _split_axis(fg: np.ndarray, axis: int, min_gap: int) -> list[tuple]:
    """Split foreground mask along an axis using white-space gaps. Axis 0
    means horizontal slices (gaps in rows → top/bottom split); axis 1 means
    vertical slices (gaps in cols → left/right split). Returns list of
    (start, end) ranges along the OPPOSITE axis covering each foreground
    block."""
    H, W = fg.shape
    projection = fg.sum(axis=axis)
    orthogonal_size = H if axis == 0 else W
    # Permit a thin rule or a few anti-aliased pixels through a whitespace
    # valley. This separates panels joined by a 1–2 px connector while
    # keeping ordinary text strokes active across the cut.
    valley_occupancy = max(2, int(round(orthogonal_size * 0.012)))
    has_fg = projection > valley_occupancy
    gaps = _find_gaps(has_fg, min_gap)
    if not gaps:
        return []
    pieces = []
    prev = 0
    for g_s, g_e in gaps:
        if g_s - prev > 0:
            pieces.append((prev, g_s))
        prev = g_e
    last = W if axis == 0 else H
    if last - prev > 0:
        pieces.append((prev, last))
    return pieces


def detect_outline_mask(cleaned: np.ndarray,
                        x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    """Return an alpha mask for a large pale rectangular card outline.

    Splitting parents along whitespace can throw away the outer 1-2 px
    rounded border of a big card because that border is sparse and split
    away from the interior content. This helper keeps only pixels near
    the raw component perimeter, so the resulting asset restores the
    border without duplicating every child icon and panel inside.

    Returns None when the candidate isn't a plausible card outline:
    too small, no edge coverage, or coverage missing on too many sides.
    Big rectangles (≥250×120 at 720-scale) get more permissive
    thresholds since a rectangle this large with even sparse edge
    coverage on 3+ sides is almost certainly a real card border.
    """
    crop = cleaned[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    scale = pixel_scale(cleaned)
    if w < s_length(180, scale) or h < s_length(80, scale):
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    diff_white = np.abs(crop.astype(int) - 255).max(axis=2)
    fg = (
        ((gray < 253) & (diff_white > 5))
        | ((hsv[:, :, 1] > 8) & (diff_white > 4))
    )
    band_lo = s_length(4, scale)
    band_hi = s_length(8, scale)
    band = max(band_lo, min(band_hi, int(round(min(w, h) * 0.03))))
    edge_zone = np.zeros((h, w), dtype=bool)
    edge_zone[:band, :] = True
    edge_zone[-band:, :] = True
    edge_zone[:, :band] = True
    edge_zone[:, -band:] = True
    edge = fg & edge_zone
    # `80` is a hard pixel floor on edge coverage; the `0.003 × w × h`
    # term is a fractional floor that already auto-scales with the
    # crop area. Only the hard floor needs scaling by area.
    if int(edge.sum()) < max(s_area(80, scale), int(0.003 * w * h)):
        return None

    top = float(np.any(fg[:band, :], axis=0).sum()) / max(1, w)
    bottom = float(np.any(fg[-band:, :], axis=0).sum()) / max(1, w)
    left = float(np.any(fg[:, :band], axis=1).sum()) / max(1, h)
    right = float(np.any(fg[:, -band:], axis=1).sum()) / max(1, h)
    big = (w >= s_length(250, scale)) and (h >= s_length(120, scale))
    side_floor = 0.22 if big else 0.28
    max_v_floor = 0.30 if big else 0.42
    max_h_floor = 0.22 if big else 0.32
    side_count = sum(v >= side_floor for v in (top, bottom, left, right))
    if side_count < 3 or max(top, bottom) < max_v_floor or max(left, right) < max_h_floor:
        return None

    alpha = (edge.astype(np.uint8) * 255)
    alpha = cv2.dilate(alpha, np.ones((2, 2), np.uint8), iterations=1)
    return alpha


def detect_outline_rects(cleaned: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    """Find nested large card outlines that a single huge component
    would otherwise hide. detect_components returns one merged CC for a
    multi-card panel; this function uses contour analysis on the same
    foreground mask to surface each card's border as its own outline
    record with a ring-shaped alpha mask.
    """
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
    diff_white = np.abs(cleaned.astype(int) - 255).max(axis=2)
    mask = (
        ((gray < 253) & (diff_white > 5))
        | ((hsv[:, :, 1] > 8) & (diff_white > 4))
    ).astype(np.uint8) * 255
    # Edge layer also feeds the contour pass so faint borders that
    # don't pass the fill threshold still get surfaced as outlines.
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 90)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.bitwise_or(mask, edges)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int, np.ndarray]] = []
    img_h, img_w = cleaned.shape[:2]
    scale = pixel_scale(cleaned)
    min_w = s_length(180, scale)
    min_h = s_length(80, scale)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_w or h < min_h:
            continue
        if w > img_w * 0.96 or h > img_h * 0.90:
            continue
        alpha = detect_outline_mask(cleaned, x, y, x + w, y + h)
        if alpha is None:
            continue
        candidates.append((int(x), int(y), int(x + w), int(y + h), alpha))

    # Prefer the tighter nested contour over an external contour that also
    # includes a tab label or adjacent connector.
    out: list[tuple[int, int, int, int, np.ndarray]] = []
    for cand in sorted(candidates,
                       key=lambda r: (r[2] - r[0]) * (r[3] - r[1])):
        x1, y1, x2, y2, _ = cand
        area = max(1, (x2 - x1) * (y2 - y1))
        duplicate = False
        for ox1, oy1, ox2, oy2, _om in out:
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
            similar_size = min(area, oarea) / max(area, oarea) >= 0.55
            if similar_size and (ix2 - ix1) * (iy2 - iy1) >= 0.85 * min(area, oarea):
                duplicate = True
                break
        if not duplicate:
            out.append(cand)
    return out


def conservative_split(crop: np.ndarray, min_gap: int = 12,
                       foreground_mask: np.ndarray | None = None) -> list[tuple]:
    """Split via white-space gaps in BOTH horizontal and vertical directions.
    Tries vertical-strip split first (left-right), then for each strip tries
    horizontal split (top-bottom). Any clean gap of `min_gap` px or more is
    accepted — there is no sub-piece size threshold. The previous
    `min_split_dim` guard removed: it suppressed legitimate splits when an
    icon was visually small but cleanly separated from its neighbour.

    Returns [(x1,y1,x2,y2), ...] relative to crop. Single-element list when
    no clean split is possible."""
    H, W = crop.shape[:2]
    # Use a shorter whitespace gap for compact cards and nested panels.
    # A fixed 12 px (scaled) gap misses the 8–12 px gutters common in
    # repeated rows, while the proportional cap keeps large regions using
    # the caller's conservative threshold.
    min_gap = min(min_gap, max(6, int(round(min(H, W) * 0.035))))
    if foreground_mask is not None and foreground_mask.shape[:2] == (H, W):
        fg = (foreground_mask > 0).astype(np.uint8)
    else:
        # Standalone fallback: compare with the crop's corner-estimated
        # canvas color instead of treating every dark-background pixel as
        # foreground (the previous gray<225 rule did exactly that).
        bg = _estimate_background_bgr(crop).astype(np.int16)
        delta = np.abs(crop.astype(np.int16) - bg).max(axis=2)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        fg = ((delta >= 8) | ((delta >= 5) & (hsv[:, :, 1] > 18))).astype(np.uint8)

    def fit_bbox(strip):
        rs = np.where(strip.sum(axis=1) > 0)[0]
        cs = np.where(strip.sum(axis=0) > 0)[0]
        if len(rs) and len(cs):
            return (int(cs[0]), int(rs[0]), int(cs[-1]) + 1, int(rs[-1]) + 1)
        return None

    # Vertical-strip split (left-right) first
    col_pieces = _split_axis(fg, axis=0, min_gap=min_gap)
    if not col_pieces:
        col_pieces = [(0, W)]

    sub_boxes = []
    for cs, ce in col_pieces:
        col_strip = fg[:, cs:ce]
        # Within each column-strip, try horizontal split (top-bottom)
        row_pieces = _split_axis(col_strip, axis=1, min_gap=min_gap)
        if not row_pieces:
            row_pieces = [(0, H)]
        for rs, re in row_pieces:
            block = col_strip[rs:re]
            tight = fit_bbox(block)
            if tight is None:
                continue
            sub_boxes.append((cs + tight[0], rs + tight[1],
                              cs + tight[2], rs + tight[3]))

    if len(sub_boxes) <= 1:
        return [(0, 0, W, H)]
    return sub_boxes


def run(*, clean: str, source: str, ocr: str, out: str,
        debug_dir: str | None = None,
        masks_dir: str | None = None,
        min_area: int = 80, dilate: int = 6, split_gap: int = 12) -> None:
    """Programmatic entry — see parse_args() for the CLI equivalent.

    Lets run_pipeline.process_page invoke build_inventory in-process,
    skipping subprocess startup cost.
    """
    args = argparse.Namespace(
        clean=clean, source=source, ocr=ocr, out=out,
        debug_dir=debug_dir, masks_dir=masks_dir,
        min_area=min_area, dilate=dilate, split_gap=split_gap,
    )
    _run(args)


def main() -> None:
    _run(parse_args())


def _run(args: argparse.Namespace) -> None:
    from inventory.builder import InventoryBuilder
    InventoryBuilder(args).build_and_write()


if __name__ == "__main__":
    main()

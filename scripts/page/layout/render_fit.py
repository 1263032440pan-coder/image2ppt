"""Fit local font family and text formatting against source ink.

Candidate families come from locally installed or explicitly configured
fonts. Pillow renders each face and style combination for comparison with
the source text mask. A missing or poor fit returns ``None`` so the caller
uses the language-specific font fallback.
"""
from __future__ import annotations

import numpy as np

from layout.render_font import (
    _font_candidates,
    _render_text_metrics,
    font_families_match,
    has_cjk_text,
)
from layout.render_ink import _mask_shape_error, _target_text_metrics
from layout.text_sizing import _FONT_LADDER
from layout.text_units import _estimated_text_width_px


def _binary_size_candidates(
    text: str,
    font_path: str,
    bold: bool,
    target_h: int,
    initial_size: int,
    pt_per_px: float,
    face_index: int = 0,
) -> list[int]:
    """Fast-lock likely pt sizes by binary-searching rendered ink height."""
    lo, hi = 5, 96
    for _ in range(8):
        if lo >= hi:
            break
        mid = (lo + hi) // 2
        metrics = _render_text_metrics(
            text, font_path, mid, pt_per_px, bold, face_index=face_index)
        mid_h = int(metrics["ink_h"]) if metrics else 0
        if mid_h < target_h:
            lo = mid + 1
        else:
            hi = mid
    center = lo
    candidates = set(range(center - 3, center + 4))
    candidates.update(range(int(initial_size) - 4, int(initial_size) + 5))
    candidates.update(v for v in _FONT_LADDER
                      if abs(v - int(initial_size)) <= 8)
    return sorted(v for v in candidates if 5 <= v <= 96)


def fit_text_render(
    text: str,
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    initial_size: int,
    initial_bold: bool,
    initial_font: str,
    pt_per_px: float,
) -> dict | None:
    """Choose font/weight/size/position by comparing local renders to ink.

    The default candidate list contains only local, renderable font files.
    Candidate faces come from the current local-font catalog and exact local
    family lookups; no classifier or image-derived family rules are consulted.
    """
    if not text or "\n" in text:
        return None
    target = _target_text_metrics(source, bbox)
    if target is None or target["ink_h"] < 3 or target["ink_w"] < 3:
        return None
    cjk_text = has_cjk_text(text)
    fonts = _font_candidates(
        initial_font,
        cjk=True if cjk_text else False,
        text=text,
    )
    if not fonts:
        return None
    best: dict | None = None
    bold_options = [False, True]
    italic_options = [False, True]

    for font in fonts:
        font_path = font["path"]
        face_index = int(font.get("index", 0) or 0)
        font_penalty = float(font.get("penalty", 0.0))
        for bold in bold_options:
            try:
                size_candidates = _binary_size_candidates(
                    text, font_path, bold, int(target["ink_h"]),
                    int(initial_size), pt_per_px, face_index=face_index,
                )
            except Exception:
                continue
            for italic in italic_options:
                for size in size_candidates:
                    try:
                        metrics = _render_text_metrics(
                            text, font_path, size, pt_per_px, bold,
                            face_index=face_index, italic=italic)
                    except Exception:
                        continue
                    if not metrics:
                        continue
                    h_err = abs(metrics["ink_h"] - target["ink_h"]) / max(
                        1, target["ink_h"])
                    w_err = abs(metrics["ink_w"] - target["ink_w"]) / max(
                        1, target["ink_w"])
                    area_err = abs(
                        metrics["ink_area"] - target["ink_area"]) / max(
                        1, target["ink_area"])
                    shape_err = _mask_shape_error(
                        target["mask"], metrics["mask"])
                    size_penalty = abs(size - int(initial_size)) / max(
                        16.0, float(initial_size))
                    visual_loss = (
                        0.40 * h_err
                        + 0.30 * w_err
                        + 0.12 * min(1.5, area_err)
                        + 0.12 * shape_err
                    )
                    prior_penalty = (
                        0.035 * min(2.0, size_penalty)
                        + font_penalty
                    )
                    score = visual_loss + prior_penalty
                    candidate = {
                        "score": float(score),
                        "visual_loss": float(visual_loss),
                        "prior_penalty": float(prior_penalty),
                        "height_err": float(h_err),
                        "width_err": float(w_err),
                        "shape_err": float(shape_err),
                        "font": font["ppt_name"],
                        "font_path": font_path,
                        "font_face_id": font.get("font_face_id"),
                        "face_index": face_index,
                        "size": int(size),
                        "bold": bool(bold),
                        "italic": bool(italic),
                        "metrics": metrics,
                        "target": target,
                    }
                    if best is None or candidate["score"] < best["score"]:
                        best = candidate

    if best is None:
        return None
    # If the best render is still far from the source ink, keep the
    # conservative bbox-height fallback rather than trusting a bad match.
    if (best["score"] > 0.55
            and (best["height_err"] > 0.35 or best["width_err"] > 0.55)):
        return None

    x1, y1, x2, y2 = (int(v) for v in bbox)
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    metrics = best["metrics"]
    target = best["target"]
    rx1, ry1, rx2, ry2 = (int(v) for v in metrics["ink_bbox"])
    tx1, ty1, tx2, ty2 = (int(v) for v in target["ink_bbox"])

    raw_x = tx1 - rx1
    raw_y = ty1 - ry1
    max_dx = max(5, int(round(0.35 * bbox_h)))
    max_dy = max(6, int(round(0.75 * bbox_h)))
    fit_x = int(round(max(x1 - max_dx, min(x1 + max_dx, raw_x))))
    fit_y = int(round(max(y1 - max_dy, min(y1 + max_dy, raw_y))))
    fit_x = max(0, min(source.shape[1] - 1, fit_x))
    fit_y = max(0, min(source.shape[0] - 1, fit_y))

    render_w = int(metrics["ink_w"])
    line_px = float(best["size"]) / max(0.01, pt_per_px)
    text_w = max(
        bbox_w + 6,
        render_w + 8,
        _estimated_text_width_px(
            text, int(best["size"]),
            pt_per_px=pt_per_px,
            bold=bool(best["bold"]),
        ) + 8,
    )
    text_w = min(int(text_w), max(1, int(source.shape[1]) - fit_x))
    text_h = max(
        bbox_h + 4,
        int(round(max(float(ry2) + 4.0, line_px * 1.35))),
    )

    return {
        "font": best["font"],
        "font_face_id": best.get("font_face_id"),
        "font_path": best["font_path"],
        "face_index": int(best.get("face_index", 0)),
        "size": int(best["size"]),
        "bold": bool(best["bold"]),
        "italic": bool(best["italic"]),
        "box": [fit_x, fit_y, int(text_w), int(text_h)],
        "score": round(float(best["score"]), 4),
        "visual_loss": round(float(best["visual_loss"]), 6),
        "prior_penalty": round(float(best["prior_penalty"]), 6),
        "target_ink": [int(tx1), int(ty1), int(tx2), int(ty2)],
        "render_ink": [int(rx1), int(ry1), int(rx2), int(ry2)],
    }


def measure_text_render_candidate(
    text: str,
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    font: str,
    size: int,
    bold: bool,
    pt_per_px: float,
    font_path: str | None = None,
    face_index: int = 0,
) -> dict | None:
    """Measure a fixed rendered style with a prior-free shared visual loss."""
    if not text or "\n" in text or pt_per_px <= 0:
        return None
    if font_path is None:
        candidates = _font_candidates(
            font, cjk=True if has_cjk_text(text) else False, text=text)
        matching = [item for item in candidates
                    if font_families_match(font, item["ppt_name"])]
        if not matching:
            return None
        font_path = matching[0]["path"]
        face_index = int(matching[0].get("index", 0) or 0)
    target = _target_text_metrics(source, bbox)
    rendered = _render_text_metrics(
        text, str(font_path), int(size), pt_per_px, bool(bold),
        face_index=int(face_index))
    if target is None or rendered is None:
        return None
    h_err = abs(rendered["ink_h"] - target["ink_h"]) / max(1, target["ink_h"])
    w_err = abs(rendered["ink_w"] - target["ink_w"]) / max(1, target["ink_w"])
    area_err = abs(rendered["ink_area"] - target["ink_area"]) / max(
        1, target["ink_area"])
    shape_err = _mask_shape_error(target["mask"], rendered["mask"])
    visual_loss = (0.40 * h_err + 0.30 * w_err
                   + 0.12 * min(1.5, area_err) + 0.12 * shape_err)
    return {
        "visual_loss": float(visual_loss),
        "height_err": float(h_err),
        "width_err": float(w_err),
        "area_err": float(area_err),
        "shape_err": float(shape_err),
        "render_ink": rendered["ink_bbox"],
        "target_ink": target["ink_bbox"],
        "font_path": str(font_path),
        "face_index": int(face_index),
    }

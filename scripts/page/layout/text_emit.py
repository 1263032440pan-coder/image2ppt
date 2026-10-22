"""Build one layout text-element dict from an inventory text entry.

This is the text branch of the per-page main loop, extracted as a
single function so the builder can stay focused on the image branch
and orchestration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]    # scripts/
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from text_safety import ppt_safe_text  # noqa: E402

from layout.bbox_shape import _trim_short_stat_text_bbox
from layout.render_font import default_ppt_font, resolve_ppt_font
from layout.render_ink import _target_text_metrics
from layout.text_mixed import mixed_size_runs_from_char_boxes
from layout.text_sizing import (
    CAPTURE_FACTOR_SHORT,
    _inherit_punct_sizes,
    _should_apply_run_font_sizes,
    _snap_to_ladder,
    _unify_run_sizes_by_color,
    font_size_pt,
)
from layout.text_style import detect_text_style
from layout.text_units import (
    _derive_source_char_boxes,
    _estimated_runs_width_px,
    _estimated_text_width_px,
)


def _char_box_source(el: dict, raw_text: str, char_boxes,
                    words, word_boxes) -> str:
    if (char_boxes is not None and len(char_boxes) == len(raw_text)
            and isinstance(el.get("chars"), list)
            and "".join(str(ch) for ch in el["chars"]) == raw_text):
        return "ocr_char_boxes"
    if (words is not None and word_boxes is not None
            and len(words) == len(word_boxes)
            and "".join(str(word) for word in words) == raw_text):
        return "word_boxes_proportional"
    return "line_bbox_proportional"


def _fit_inline_runs(runs: list[dict], text: str, char_boxes,
                     source: np.ndarray, fit_text_render,
                     pt_per_px: float, fallback_size: int) -> None:
    """Fit each sized run against local font renders, preserving its bbox."""
    if (not char_boxes or len(char_boxes) != len(text)
            or "".join(str(run.get("text") or "") for run in runs) != text):
        return
    offset = 0
    for run in runs:
        run_text = str(run.get("text") or "")
        start, end = offset, offset + len(run_text)
        offset = end
        boxes = char_boxes[start:end]
        if not run_text.strip() or len(boxes) != len(run_text) or not boxes:
            continue
        try:
            x1 = min(int(box[0]) for box in boxes)
            y1 = min(int(box[1]) for box in boxes)
            x2 = max(int(box[2]) for box in boxes)
            y2 = max(int(box[3]) for box in boxes)
        except (TypeError, ValueError, IndexError):
            continue
        try:
            fit = fit_text_render(
                run_text, source, (x1, y1, x2, y2),
                initial_size=int(run.get("size") or fallback_size),
                initial_bold=False,
                initial_font=default_ppt_font(run_text),
                pt_per_px=pt_per_px,
            )
        except Exception as exc:
            fit = None
            run["font_fit_fallback_reason"] = (
                f"render_exception:{type(exc).__name__}")
        if fit is None:
            run["font"] = default_ppt_font(run_text)
            run["bold"] = False
            run["italic"] = False
            run.setdefault("font_fit_fallback_reason",
                           "no_usable_local_render_fit")
            continue
        run["font"] = resolve_ppt_font(fit["font"], text=run_text)
        run["size"] = int(fit["size"])
        run["bold"] = bool(fit["bold"])
        run["italic"] = bool(fit["italic"])
        run["font_fit_score"] = fit["score"]


def _emit_text_element_record(
    el: dict, x1: int, y1: int, x2: int, y2: int,
    source: np.ndarray, pt_per_px: float,
) -> dict | None:
    """Build the layout dict for one inventory `text` entry.

    Returns the record dict, or None when the entry has no visible text
    to emit.
    """
    if not str(el.get("text", "") or "").strip():
        return None
    # Look up fit_text_render through the facade so tests can patch it
    # (`itl.fit_text_render = fake_fit`).
    import inventory_to_layout as _itl
    fit_text_render = _itl.fit_text_render

    x1, y1, x2, y2 = _trim_short_stat_text_bbox(
        str(el.get("text", "") or ""),
        (int(x1), int(y1), int(x2), int(y2)),
        source,
    )
    # PaddleOCR per-char bboxes when present. ocr_paddle attaches these
    # only when len(chars) == len(text); re-verify against the current
    # text (OCR-review may have corrected it).
    raw_text = el["text"]
    safe_text = ppt_safe_text(raw_text)
    cb = el.get("char_boxes")
    if cb is not None and len(cb) != len(raw_text):
        cb = None
    wd = el.get("words")
    wb = el.get("word_boxes")
    if (wd is not None and wb is not None
            and (len(wd) != len(wb) or "".join(wd) != raw_text)):
        wd = wb = None
    source_char_boxes = _derive_source_char_boxes(
        raw_text, (int(x1), int(y1), int(x2), int(y2)), cb, wd, wb)
    char_box_source = _char_box_source(el, raw_text, cb, wd, wb)
    style = detect_text_style([x1, y1, x2, y2], source,
                              text=raw_text,
                              char_boxes=source_char_boxes,
                              words=wd, word_boxes=wb)
    apply_run_sizes = _should_apply_run_font_sizes(raw_text)
    if "runs" in style:
        for r in style["runs"]:
            gh = r.pop("glyph_h", None)
            if apply_run_sizes and gh and gh > 0:
                raw = gh * pt_per_px * CAPTURE_FACTOR_SHORT
                r["size"] = _snap_to_ladder(max(8.0, min(36.0, raw)))
        if apply_run_sizes:
            # Same-colour runs share a font size; punctuation inherits
            # its neighbour's em-box.
            _unify_run_sizes_by_color(style["runs"])
            _inherit_punct_sizes(style["runs"])
    mixed_size = None
    if not apply_run_sizes:
        mixed_size = mixed_size_runs_from_char_boxes(
            raw_text, source, (int(x1), int(y1), int(x2), int(y2)),
            source_char_boxes, style.get("runs"), style["color"],
            pt_per_px=pt_per_px,
        )
        if mixed_size is not None:
            style["runs"] = mixed_size["runs"]
    if "runs" in style:
        for r in style["runs"]:
            r["text"] = ppt_safe_text(r.get("text", ""))
    # Decoration-padding probe: when a multi-char line's ink vertical
    # extent is well under bbox_h (≤ 50 %), the bbox likely includes
    # shadow/glow above and below the actual glyphs. Use ink_h.
    ink_h = style.pop("ink_h", None)
    effective_h = int(y2 - y1)
    visible_chars = sum(1 for c in raw_text if c.strip() and c != "\n")
    if (ink_h and effective_h > 30 and visible_chars >= 3
            and ink_h <= 0.5 * effective_h):
        effective_h = ink_h
    size = font_size_pt(safe_text, x2 - x1, effective_h,
                        pt_per_px=pt_per_px)
    if mixed_size is not None:
        size = max(int(size), int(mixed_size["base_size"]))
    ppt_font = default_ppt_font(safe_text)
    text_bold = False
    text_italic = False
    run_sized = any(r.get("size") is not None
                    for r in style.get("runs", []))
    render_fit = None
    render_fit_error = None
    if not run_sized:
        try:
            render_fit = fit_text_render(
                safe_text, source, (int(x1), int(y1), int(x2), int(y2)),
                initial_size=int(size), initial_bold=False,
                initial_font=ppt_font, pt_per_px=pt_per_px,
            )
        except Exception as exc:
            render_fit_error = f"render_exception:{type(exc).__name__}"
    if render_fit is not None:
        ppt_font = resolve_ppt_font(render_fit["font"], text=safe_text)
        size = int(render_fit["size"])
        text_bold = bool(render_fit["bold"])
        text_italic = bool(render_fit["italic"])
    elif run_sized and style.get("runs"):
        _fit_inline_runs(
            style["runs"], safe_text, source_char_boxes, source,
            fit_text_render, pt_per_px, int(size))
        ppt_font = str(style["runs"][0].get("font") or ppt_font)
    # Tight bbox: do NOT inflate the text box beyond what OCR found.
    # 6 px right + 2 px bottom safety margin avoids font-metric clip.
    align = str(el.get("align") or "left")
    base_w = int(x2 - x1)
    text_w = max(
        base_w + 6,
        _estimated_text_width_px(
            safe_text, int(size),
            pt_per_px=pt_per_px, bold=text_bold) + 8,
    )
    if run_sized:
        text_w = max(
            text_w,
            _estimated_runs_width_px(
                style.get("runs", []), int(size),
                pt_per_px=pt_per_px, bold=text_bold) + 8,
        )
    text_h = int(y2 - y1) + 2
    if run_sized:
        text_h = max(
            text_h,
            int(round((int(size) / max(0.01, pt_per_px)) * 1.20)),
        )
    text_x = int(x1)
    text_y = int(y1)
    valign_mode = "middle"
    if render_fit is not None:
        text_x, text_y, text_w, text_h = render_fit["box"]
        # render_fit picks a tight box around the glyph metrics; with
        # valign=top in PowerPoint/LibreOffice the ascent line is pinned to
        # the box top, which makes the rendered text sit ~half the line
        # leading above where the OCR found it (especially obvious on
        # short banner-style titles like "① 数据收集"). Center vertically
        # inside the render_fit box instead — the box height already
        # tracks the glyph + a small padding, so MIDDLE puts the ink right
        # back on the source y-axis.
        valign_mode = "middle"
    elif mixed_size is not None:
        valign_mode = "top"
    text_w = min(int(text_w),
                 max(1, int(source.shape[1]) - int(text_x)))
    record = {
        "type": "text", "name": el["id"], "text": safe_text,
        "box": [text_x, text_y, text_w, text_h],
        "source_bbox": [int(x1), int(y1), int(x2), int(y2)],
        "font": ppt_font, "size": int(size),
        "bold": text_bold, "italic": text_italic,
        "color": style["color"], "align": align, "valign": valign_mode,
        "align_source": ("inventory" if el.get("align")
                         else "single_line_anchor"),
    }
    if (source_char_boxes is not None
            and len(source_char_boxes) == len(raw_text)
            and len(safe_text) == len(raw_text)):
        record["source_chars"] = list(safe_text)
        record["source_char_boxes"] = [
            [int(v) for v in box] for box in source_char_boxes
        ]
    target_metrics = _target_text_metrics(
        source, (int(x1), int(y1), int(x2), int(y2)))
    if target_metrics is not None:
        record["target_ink"] = target_metrics["ink_bbox"]
    if render_fit is not None:
        record["size_source"] = "render_fit"
        record["font_fit_score"] = render_fit["score"]
        record["render_fit_font"] = render_fit["font"]
        record["fit_target_ink"] = render_fit["target_ink"]
        record["fit_render_ink"] = render_fit["render_ink"]
    elif mixed_size is not None:
        record["size_source"] = "mixed_runs"
        record["mixed_size"] = {
            "base_size": mixed_size["base_size"],
            "suffix_size": mixed_size["suffix_size"],
            "prefix_h": mixed_size["prefix_h"],
            "suffix_h": mixed_size["suffix_h"],
            "prefix_bold": mixed_size.get("prefix_bold"),
            "suffix_bold": mixed_size.get("suffix_bold"),
            "prefix_fit_score": mixed_size["prefix_fit_score"],
            "suffix_fit_score": mixed_size["suffix_fit_score"],
        }
    run_fit_success = any(
        run.get("font_fit_score") is not None
        for run in style.get("runs", []))
    record["font_fit_source"] = (
        "local_render_fit" if render_fit is not None else
        "local_render_fit_runs" if run_fit_success else
        "script_fallback")
    if render_fit is None and not run_fit_success:
        record["font_fit_fallback_reason"] = (
            render_fit_error or "no_usable_local_render_fit")
    # Optional in-bbox per-character color runs.
    if "runs" in style:
        record["runs"] = style["runs"]
    return record

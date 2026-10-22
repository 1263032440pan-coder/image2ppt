"""Conservatively group adjacent OCR rows into editable paragraphs.

Rows are merged only when their source geometry and recovered text style
strongly agree. Every merged row retains its own paragraph geometry and
style in the layout record so the PPTX writer can reproduce indentation and
line spacing without moving uncertain rows.
"""
from __future__ import annotations

import math
import re
from statistics import median
from typing import Any


_FONT_KEY_RE = re.compile(r"[^a-z0-9\u3400-\u9fff]+")


def _font_key(value: Any) -> str:
    return _FONT_KEY_RE.sub("", str(value or "").casefold())


def _box(el: dict, key: str) -> tuple[float, float, float, float] | None:
    value = el.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if key == "box":
        x2, y2 = x + w, y + h
    else:
        x2, y2 = w, h
    if not all(math.isfinite(v) for v in (x, y, x2, y2)) or x2 <= x or y2 <= y:
        return None
    return x, y, x2, y2


def _style_compatible(a: dict, b: dict) -> bool:
    if _font_key(a.get("font")) != _font_key(b.get("font")):
        return False
    if bool(a.get("bold")) != bool(b.get("bold")):
        return False
    if bool(a.get("italic")) != bool(b.get("italic")):
        return False
    try:
        size_a, size_b = float(a.get("size")), float(b.get("size"))
    except (TypeError, ValueError):
        return False
    if abs(size_a - size_b) > max(2.5, 0.15 * max(size_a, size_b)):
        return False
    ca, cb = a.get("color"), b.get("color")
    if ca == cb:
        return True
    try:
        def channels(value: Any) -> tuple[int, int, int]:
            text = str(value).lstrip("#")
            if len(text) != 6:
                raise ValueError
            return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
        rgb_a, rgb_b = channels(ca), channels(cb)
    except (TypeError, ValueError):
        return False
    return max(abs(x - y) for x, y in zip(rgb_a, rgb_b)) <= 18


def _eligible_pair(a: dict, b: dict) -> tuple[bool, float]:
    text_a = str(a.get("text") or "").strip()
    text_b = str(b.get("text") or "").strip()
    if (len(text_a) < 4 or len(text_b) < 4 or "\n" in text_a or "\n" in text_b
            or a.get("list") or b.get("list")
            or a.get("ignored_marker_box") or b.get("ignored_marker_box")):
        return False, 0.0
    if not _style_compatible(a, b):
        return False, 0.0
    ba, bb = _box(a, "source_bbox"), _box(b, "source_bbox")
    if ba is None or bb is None or bb[1] <= ba[1]:
        return False, 0.0
    ha, hb = ba[3] - ba[1], bb[3] - bb[1]
    line_h = max(1.0, median((ha, hb)))
    gap = bb[1] - ba[3]
    if gap < -0.18 * line_h or gap > 0.82 * line_h:
        return False, 0.0
    dx = abs(bb[0] - ba[0])
    if dx > max(5.0, 0.62 * line_h):
        return False, 0.0
    center_delta = abs(((bb[1] + bb[3]) / 2.0) - ((ba[1] + ba[3]) / 2.0))
    if center_delta > 1.95 * line_h:
        return False, 0.0
    score = (max(0.0, gap) / line_h) + 0.85 * (dx / line_h)
    return True, score


def _alignment(lines: list[dict]) -> str:
    boxes = [_box(line, "source_bbox") for line in lines]
    boxes = [box for box in boxes if box is not None]
    if len(boxes) < 2:
        return "left"
    heights = [box[3] - box[1] for box in boxes]
    tolerance = max(3.0, 0.18 * median(heights))
    lefts = [box[0] for box in boxes]
    rights = [box[2] for box in boxes]
    centers = [(box[0] + box[2]) / 2.0 for box in boxes]
    if max(lefts) - min(lefts) <= tolerance:
        return "left"
    if max(rights) - min(rights) <= tolerance:
        return "right"
    if max(centers) - min(centers) <= tolerance:
        return "center"
    return "left"


def _line_runs(line: dict) -> list[dict]:
    text = str(line.get("text") or "")
    runs = line.get("runs")
    if runs and "".join(str(run.get("text") or "") for run in runs) == text:
        out = []
        for run in runs:
            if not str(run.get("text") or ""):
                continue
            item = dict(run)
            for key in ("font", "size", "bold", "italic", "color"):
                if item.get(key) is None and line.get(key) is not None:
                    item[key] = line[key]
            if (item.get("font_fit_score") is None
                    and line.get("font_fit_score") is not None):
                item["font_fit_score"] = line["font_fit_score"]
            out.append(item)
        return out
    return [{
        "text": text,
        "font": line.get("font"),
        "size": line.get("size"),
        "bold": line.get("bold"),
        "italic": line.get("italic"),
        "color": line.get("color"),
        "font_fit_score": line.get("font_fit_score"),
    }]


def _paragraph_properties(lines: list[dict], pt_per_px: float) -> list[dict]:
    output_boxes = [_box(line, "box") for line in lines]
    source_boxes = [_box(line, "source_bbox") for line in lines]
    if any(box is None for box in output_boxes + source_boxes):
        return []
    x_min = min(box[0] for box in output_boxes if box)
    align_value = _alignment(lines)
    source_lefts = [box[0] for box in source_boxes if box]
    baseline_left = median(source_lefts[1:] or source_lefts)
    font_sizes = [max(1.0, float(line.get("size") or 1.0)) for line in lines]
    valid_scale = max(0.01, float(pt_per_px))
    paragraphs: list[dict] = []
    baselines: list[float] = []
    for index, (line, out_box, src_box) in enumerate(
            zip(lines, output_boxes, source_boxes)):
        baseline = float(src_box[3])
        em_height_px = font_sizes[index] / valid_scale
        props: dict[str, Any] = {
            "source_line_name": line.get("name"),
            "align": align_value,
            "line_spacing_source": "ocr_baseline_distance",
            "spacing_confidence": 0.82,
            "left_indent_px": 0.0,
            "first_line_indent_px": 0.0,
            "space_before_pt": None,
            "space_after_pt": None,
        }
        if align_value == "left":
            base_indent = max(0.0, float(baseline_left) - x_min)
            props["left_indent_px"] = round(base_indent, 2)
            if index == 0:
                props["first_line_indent_px"] = round(
                    float(out_box[0] - x_min) - base_indent, 2)
            else:
                props["left_indent_px"] = round(
                    max(0.0, float(out_box[0] - x_min)), 2)
        props["line_spacing"] = 1.0
        paragraphs.append(props)
        baselines.append(baseline)
    for index in range(len(paragraphs) - 1):
        baseline_delta = baselines[index + 1] - baselines[index]
        ratio = baseline_delta / max(1.0, font_sizes[index] / valid_scale)
        paragraphs[index]["line_spacing"] = round(
            max(0.75, min(2.4, ratio)), 3)
    return paragraphs


def _merge_group(lines: list[dict], pt_per_px: float) -> dict:
    output_boxes = [_box(line, "box") for line in lines]
    source_boxes = [_box(line, "source_bbox") for line in lines]
    x1 = min(box[0] for box in output_boxes if box)
    y1 = min(box[1] for box in output_boxes if box)
    x2 = max(box[2] for box in output_boxes if box)
    y2 = max(box[3] for box in output_boxes if box)
    sx1 = min(box[0] for box in source_boxes if box)
    sy1 = min(box[1] for box in source_boxes if box)
    sx2 = max(box[2] for box in source_boxes if box)
    sy2 = max(box[3] for box in source_boxes if box)
    paragraphs = _paragraph_properties(lines, pt_per_px)
    runs: list[dict] = []
    for index, line in enumerate(lines):
        if index:
            runs.append({"text": "\n"})
        runs.extend(_line_runs(line))
    merged = dict(lines[0])
    merged.update({
        "text": "\n".join(str(line.get("text") or "") for line in lines),
        "box": [int(round(x1)), int(round(y1)),
                max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1)))],
        "source_bbox": [int(round(sx1)), int(round(sy1)),
                        int(round(sx2)), int(round(sy2))],
        "runs": runs,
        "paragraphs": paragraphs,
        "paragraph_id": f"paragraph:{lines[0].get('name', 'text')}",
        "paragraph_source_names": [line.get("name") for line in lines],
        "paragraph_confidence": round(min(
            0.98, 0.82 + 0.04 * (len(lines) - 2)), 3),
        "paragraph_merge_reason": "adjacent_geometry_and_style_match",
        "paragraph_line_diagnostics": [
            {"name": line.get("name"),
             "text": line.get("text"),
             "source_bbox": line.get("source_bbox"),
             "font": line.get("font"), "size": line.get("size"),
             "bold": line.get("bold"), "color": line.get("color"),
             "italic": line.get("italic"),
             "font_fit_score": line.get("font_fit_score")}
            for line in lines
        ],
        "align": paragraphs[0].get("align", "left") if paragraphs else "left",
        "align_source": "ocr_paragraph_geometry" if paragraphs else "single_line_anchor",
        "paragraph_reconstruction": "merged",
    })
    for field in ("source_chars",
                  "source_char_boxes", "target_ink", "fit_target_ink",
                  "render_fit_font", "font_fit_score", "mixed_size"):
        merged.pop(field, None)
    merged.pop("line_spacing", None)
    return merged


def reconstruct_paragraphs(records: list[dict], pt_per_px: float) -> list[dict]:
    """Merge only unambiguous adjacent OCR rows; annotate untouched rows."""
    eligible = [
        index for index, record in enumerate(records)
        if _box(record, "source_bbox") is not None
        and _box(record, "box") is not None
        and str(record.get("text") or "").strip()
        and "\n" not in str(record.get("text") or "")
    ]
    ordered = sorted(eligible, key=lambda i: (
        _box(records[i], "source_bbox")[1],
        _box(records[i], "source_bbox")[0]))
    candidate_edges: list[tuple[float, int, int]] = []
    for position, current in enumerate(ordered):
        candidates: list[tuple[float, int]] = []
        for following in ordered[position + 1:]:
            a = _box(records[current], "source_bbox")
            b = _box(records[following], "source_bbox")
            if b[1] - a[1] > 2.8 * max(a[3] - a[1], b[3] - b[1]):
                break
            ok, score = _eligible_pair(records[current], records[following])
            if ok:
                candidates.append((score, following))
        if candidates:
            candidates.sort()
            if len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= 0.18:
                score, following = candidates[0]
                candidate_edges.append((score, current, following))

    incoming: dict[int, list[tuple[float, int]]] = {}
    for score, source, target in candidate_edges:
        incoming.setdefault(target, []).append((score, source))
    best_incoming: dict[int, int] = {}
    for target, candidates in incoming.items():
        candidates.sort()
        if (len(candidates) == 1
                or candidates[1][0] - candidates[0][0] >= 0.18):
            best_incoming[target] = candidates[0][1]
    successors = {
        source: target for _score, source, target in candidate_edges
        if best_incoming.get(target) == source
    }

    predecessors = {target: source for source, target in successors.items()}
    consumed: set[int] = set()
    groups: dict[int, list[int]] = {}
    for index in ordered:
        if index in predecessors or index in consumed:
            continue
        group = [index]
        current = index
        while current in successors:
            following = successors[current]
            if following in consumed or following in group:
                break
            group.append(following)
            current = following
        if len(group) > 1:
            groups[index] = group
            consumed.update(group)

    if not groups:
        for record in records:
            record.setdefault("align_source", "single_line_anchor")
            record.setdefault("paragraph_reconstruction", "unmerged_insufficient_evidence")
        return records

    index_to_group = {member: (first, group)
                      for first, group in groups.items() for member in group}
    output: list[dict] = []
    emitted_groups: set[int] = set()
    for index, record in enumerate(records):
        group_info = index_to_group.get(index)
        if group_info is None:
            record.setdefault("align_source", "single_line_anchor")
            record.setdefault("paragraph_reconstruction", "unmerged_insufficient_evidence")
            output.append(record)
            continue
        first, members = group_info
        if first in emitted_groups:
            continue
        emitted_groups.add(first)
        output.append(_merge_group([records[item] for item in members], pt_per_px))
    return output

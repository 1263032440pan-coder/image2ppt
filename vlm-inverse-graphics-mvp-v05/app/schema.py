"""Tolerant and bounded Scene IR validation. LLM output is treated as untrusted data."""
from __future__ import annotations
import copy
import re
from typing import Any

KINDS = {"sun", "mountain", "pond", "bridge", "house", "pine", "koi", "lotus",
         "lily_pad", "reed", "rock", "ripple", "bush", "cloud"}
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def number(value: Any, default: float, low: float, high: float) -> float:
    try:
        result = float(value)
        if result != result or result in (float("inf"), float("-inf")):
            return default
        return round(max(low, min(high, result)), 3)
    except (TypeError, ValueError, OverflowError):
        return default


def color(value: Any, default: str = "#657F78") -> str:
    return value.upper() if isinstance(value, str) and _HEX.fullmatch(value) else default


def normalize_object(raw: dict, index: int = 0) -> dict | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("type", "")).lower().strip()
    if kind not in KINDS:
        return None
    bbox = raw.get("bbox", [200, 200, 300, 200])
    if not isinstance(bbox, list) or len(bbox) != 4:
        bbox = [200, 200, 300, 200]
    bbox = [number(bbox[0], 200, -100, 1100), number(bbox[1], 200, -100, 1100),
            number(bbox[2], 300, 5, 1100), number(bbox[3], 200, 5, 1100)]
    params = raw.get("params", {})
    if not isinstance(params, dict):
        params = {}
    # No scripts, SVG tags, URLs, raw paths or arbitrary attributes from model output.
    safe_params = {
        "rotation": number(params.get("rotation"), 0, -180, 180),
        "density": int(number(params.get("density"), 10, 1, 30)),
        "seed": int(number(params.get("seed"), index + 17, 0, 999999)),
        "arch": number(params.get("arch"), 0.33, 0.05, 0.85),
        "facing": "right" if params.get("facing") == "right" else "left",
        "pattern": (str(params.get("pattern", "red_white")) if
                    params.get("pattern") in ("red_white", "gold", "black", "white")
                    else "red_white"),
    }
    hint = raw.get("geometry_hint", {})
    hint = hint if isinstance(hint, dict) else {}
    def safe_points(value):
        if not isinstance(value, list):
            return []
        pts = []
        for p in value[:12]:
            if (isinstance(p, (list, tuple)) and len(p) == 2):
                try:
                    from math import isfinite
                    if not all(isfinite(float(v)) for v in p):
                        continue
                    pts.append([number(p[0], 500, 0, 1000), number(p[1], 500, 0, 1000)])
                except (ValueError, TypeError, OverflowError):
                    continue
        return pts
    mode = hint.get('mode', 'auto')
    if mode not in ('auto', 'trace', 'template'):
        mode = 'auto'
    return {
        "geometry_hint": {"mode": mode,
                          "foreground_points": safe_points(hint.get('foreground_points')),
                          "background_points": safe_points(hint.get('background_points'))},
        "id": re.sub(r"[^a-zA-Z0-9_-]", "_", str(raw.get("id", f"{kind}_{index}")))[:48] or f"obj_{index}",
        "type": kind,
        "bbox": bbox,
        "fill": color(raw.get("fill"), "#7D9990"),
        "stroke": color(raw.get("stroke"), "#283237"),
        "layer": int(number(raw.get("layer"), index, 0, 100)),
        "params": safe_params,
    }


def normalize_scene(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Scene IR must be a JSON object")
    objects = raw.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("Scene IR objects must be an array")
    validated, seen = [], set()
    for i, item in enumerate(objects[:80]):
        obj = normalize_object(item, i)
        if obj is None:
            continue
        if obj["id"] in seen:
            obj["id"] += f"_{i}"
        seen.add(obj["id"])
        validated.append(obj)
    if not validated:
        raise ValueError("No supported objects found in model Scene IR")
    validated.sort(key=lambda o: (o["layer"], o["id"]))
    return {"version": "0.1", "canvas": {"width": 1000, "height": 1000},
            "background": color(raw.get("background"), "#EFE5C4"),
            "objects": validated}


def apply_patches(scene: dict, patches: list) -> tuple[dict, list[str]]:
    """Only allow declarative parameter edits to known objects, never arbitrary code."""
    updated = copy.deepcopy(scene)
    notes = []
    objects = {o["id"]: o for o in updated["objects"]}
    if not isinstance(patches, list):
        return updated, notes
    for patch in patches[:8]:
        if not isinstance(patch, dict) or patch.get("id") not in objects:
            continue
        current = objects[patch["id"]]
        before = copy.deepcopy(current)
        candidate = {**current}
        for key in ("bbox", "fill", "stroke", "layer", "geometry_hint"):
            if key in patch:
                candidate[key] = patch[key]
        if isinstance(patch.get("params"), dict):
            candidate["params"] = {**current["params"], **patch["params"]}
        safe = normalize_object(candidate)
        if safe:
            current.update(safe)
            if before != current:
                notes.append(current["id"])
    updated["objects"].sort(key=lambda o: (o["layer"], o["id"]))
    return updated, notes

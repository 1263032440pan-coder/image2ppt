"""Font discovery + Pillow loading + per-text rasterised metrics cache."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]    # scripts/
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from fontconfig_helper import (  # noqa: E402
    font_families_match,
    fontconfig_font_path,
)


_FONT_CANDIDATES_CACHE: list[dict] | None = None
_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
_TEXT_RENDER_CACHE: dict[tuple[str, str, int, bool], dict | None] = {}

_CJK_FONT_PREFERENCE = (
    "Microsoft YaHei", "Microsoft YaHei UI",
    "Alibaba PuHuiTi", "Alibaba PuHuiTi 2.0",
    "Noto Sans CJK SC", "Noto Sans SC",
    "SimHei", "SimSun", "KaiTi", "FangSong",
    "PingFang SC", "Hiragino Sans GB", "STHeiti", "Songti",
    "Noto Serif CJK SC", "Noto Serif SC",
)
_LATIN_FONT_PREFERENCE = (
    "Arial", "Aptos", "Calibri", "Helvetica", "Tahoma", "Verdana",
    "Times New Roman", "Cambria", "Consolas", "Corbel",
)
_CJK_FAMILY_MARKERS = (
    "yahei", "puhuiti", "notosanssc", "notosanscjk",
    "notoserifsc", "notoserifcjk", "arialunicodems", "simhei",
    "simsun", "kaiti", "fangsong", "pingfang", "hiragino", "stheiti",
    "songti", "heiti",
)


def has_cjk_text(text: str) -> bool:
    return any(
        "\u2e80" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        or "\u3040" <= char <= "\u30ff"
        or "\uac00" <= char <= "\ud7af"
        for char in str(text or "")
    )


def is_cjk_font_family(family: str) -> bool:
    compact = "".join(ch for ch in str(family or "").casefold()
                      if ch.isalnum())
    return any(marker in compact for marker in _CJK_FAMILY_MARKERS)


def default_ppt_font() -> str:
    """Return a CJK family that is actually installed on this machine."""
    for family in _CJK_FONT_PREFERENCE:
        path = fontconfig_font_path(f"{family}:style=Regular")
        if not path:
            continue
        try:
            actual_family, _style = ImageFont.truetype(path, 16).getname()
        except OSError:
            continue
        if actual_family and is_cjk_font_family(actual_family):
            return actual_family
    # Keep the historical default as the last-resort name for systems where
    # font metadata cannot be queried; the PPTX remains editable there.
    return "Microsoft YaHei"


def _font_candidates(preferred_family: str | None = None,
                     *, cjk: bool | None = None) -> list[dict]:
    """Return installed fonts with Office family names and renderable files.

    ``fc-match`` is not present on normal Windows installations. The shared
    font lookup falls back to Windows' installed-font registry, so the same
    visual-fit path works on Windows, macOS and Linux.
    """
    global _FONT_CANDIDATES_CACHE
    if _FONT_CANDIDATES_CACHE is not None:
        candidates = list(_FONT_CANDIDATES_CACHE)
    else:
        names = list(dict.fromkeys((
            *_CJK_FONT_PREFERENCE,
            default_ppt_font(),
            *_LATIN_FONT_PREFERENCE,
        )))
        out: list[dict] = []
        seen_families: set[str] = set()
        for name in names:
            path = fontconfig_font_path(f"{name}:style=Regular")
            if not path:
                continue
            try:
                actual_family, _style = ImageFont.truetype(path, 16).getname()
            except OSError:
                continue
            family_key = "".join(ch for ch in actual_family.casefold()
                                 if ch.isalnum())
            if not actual_family or family_key in seen_families:
                continue
            seen_families.add(family_key)
            out.append({
                "ppt_name": actual_family,
                "path": path,
                "cjk": is_cjk_font_family(actual_family),
                "penalty": len(out) * 0.005,
            })
        _FONT_CANDIDATES_CACHE = out
        candidates = list(out)

    if cjk is not None:
        filtered = [item for item in candidates if item["cjk"] is cjk]
        # If no font was positively identified as CJK, preserve the available
        # local list instead of leaving the layout without a fit candidate.
        if filtered:
            candidates = filtered
    if preferred_family:
        candidates.sort(key=lambda item: (
            0 if font_families_match(preferred_family, item["ppt_name"]) else 1,
            float(item.get("penalty", 0.0)),
        ))
    return candidates


def resolve_ppt_font(family: str | None, *, text: str = "") -> str:
    """Resolve a predicted family to an installed, PowerPoint-visible name."""
    requested = str(family or "").strip()
    cjk = has_cjk_text(text) or is_cjk_font_family(requested)
    # Honor any installed requested family first, including fonts outside the
    # classifier's preferred list. fontconfig_font_path validates the file's
    # internal family name, so an unrelated fallback cannot pass this check.
    if requested:
        path = fontconfig_font_path(f"{requested}:style=Regular")
        if path:
            try:
                actual_family, _style = ImageFont.truetype(path, 16).getname()
            except OSError:
                actual_family = ""
            if actual_family and font_families_match(requested, actual_family):
                return actual_family

    candidates = _font_candidates(cjk=cjk)
    for candidate in candidates:
        if requested and font_families_match(requested, candidate["ppt_name"]):
            return candidate["ppt_name"]
    if candidates:
        return candidates[0]["ppt_name"]
    return requested or "Microsoft YaHei"


def _load_font(path: str, pixel_size: int) -> ImageFont.FreeTypeFont | None:
    pixel_size = max(1, int(pixel_size))
    key = (path, pixel_size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(path, pixel_size)
        except OSError:
            return None
    return _FONT_CACHE[key]


def _render_text_metrics(text: str, font_path: str, size_pt: int,
                         pt_per_px: float, bold: bool) -> dict | None:
    """Render one text line locally and return its ink bbox in source px.

    The layout stores font size in points, but the source image is in
    pixels. ``pt_per_px`` converts PPT points back to source pixels so
    the rendered mask can be compared directly with the OCR crop.
    """
    if not text or "\n" in text or pt_per_px <= 0:
        return None
    pixel_size = max(1, int(round(float(size_pt) / pt_per_px)))
    cache_key = (text, font_path, pixel_size, bool(bold))
    if cache_key in _TEXT_RENDER_CACHE:
        return _TEXT_RENDER_CACHE[cache_key]

    font = _load_font(font_path, pixel_size)
    if font is None:
        _TEXT_RENDER_CACHE[cache_key] = None
        return None

    probe = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(probe)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except UnicodeEncodeError:
        _TEXT_RENDER_CACHE[cache_key] = None
        return None
    if not bbox:
        _TEXT_RENDER_CACHE[cache_key] = None
        return None

    bx1, by1, bx2, by2 = (int(v) for v in bbox)
    pad = max(4, pixel_size // 3)
    canvas_w = max(2, bx2 - bx1 + pad * 2 + (1 if bold else 0))
    canvas_h = max(2, by2 - by1 + pad * 2)
    origin = (pad - bx1, pad - by1)
    img = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(img)
    draw.text(origin, text, font=font, fill=255)
    if bold:
        # Synthetic one-pixel emboldening — approximates how PPT/LO
        # fattens regular CJK fonts when no true bold face is available.
        draw.text((origin[0] + 1, origin[1]), text, font=font, fill=255)

    arr = np.array(img)
    ys, xs = np.where(arr > 16)
    if ys.size == 0 or xs.size == 0:
        _TEXT_RENDER_CACHE[cache_key] = None
        return None
    x_min = int(xs.min()); x_max = int(xs.max()) + 1
    y_min = int(ys.min()); y_max = int(ys.max()) + 1
    rel_bbox = [
        int(x_min - origin[0]), int(y_min - origin[1]),
        int(x_max - origin[0]), int(y_max - origin[1]),
    ]
    mask_crop = arr[y_min:y_max, x_min:x_max] > 16
    metrics = {
        "pixel_size": pixel_size,
        "ink_bbox": rel_bbox,
        "ink_w": int(x_max - x_min),
        "ink_h": int(y_max - y_min),
        "ink_area": int(mask_crop.sum()),
        "mask": mask_crop,
    }
    _TEXT_RENDER_CACHE[cache_key] = metrics
    return metrics

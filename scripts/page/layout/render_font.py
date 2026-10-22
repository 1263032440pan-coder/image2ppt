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
_FONT_CACHE: dict[tuple[str, int, int], ImageFont.FreeTypeFont] = {}
_TEXT_RENDER_CACHE: dict[tuple[str, str, int, int, bool, bool], dict | None] = {}

_LATIN_FONT_PREFERENCE = (
    "Arial", "Aptos", "Calibri", "Helvetica", "Tahoma", "Verdana",
    "Times New Roman", "Cambria", "Consolas", "Corbel",
)
_CJK_FAMILY_MARKERS = (
    "yahei", "notosanssc", "notosanscjk",
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


def fallback_ppt_font(text: str = "") -> str:
    """Return the requested script-specific fallback family name."""
    return "SimHei" if has_cjk_text(text) else "Times New Roman"


def default_ppt_font(text: str = "") -> str:
    """Return the explicit fallback used when local render fitting fails."""
    return fallback_ppt_font(text)


def _font_candidates(preferred_family: str | None = None,
                     *, cjk: bool | None = None,
                     text: str = "") -> list[dict]:
    """Return locally available regular faces eligible for raster fitting.

    CJK candidates come from the local font catalog, which only admits
    installed fonts or configured local fonts with matching permission.
    Latin candidates are resolved by exact family name on this machine.
    No classifier labels or image-derived font rules enter this list.
    """
    global _FONT_CANDIDATES_CACHE
    if _FONT_CANDIDATES_CACHE is not None:
        candidates = list(_FONT_CANDIDATES_CACHE)
    else:
        out: list[dict] = []
        seen_faces: set[tuple[str, str, int]] = set()
        try:
            from layout.font_catalog import get_font_faces
            catalog_faces = get_font_faces()
        except Exception:
            catalog_faces = []
        for face in catalog_faces:
            if not face.get("local_match_allowed"):
                continue
            style = str(face.get("style_name") or "").casefold()
            if any(marker in style for marker in ("bold", "black", "italic", "oblique")):
                continue
            path = str(face.get("font_file_path") or "")
            family = str(face.get("ppt_font_name") or face.get("font_family_name") or "")
            index = int(face.get("face_index", 0) or 0)
            if not path or not family:
                continue
            marker = (family.casefold(), path.casefold(), index)
            if marker in seen_faces:
                continue
            seen_faces.add(marker)
            out.append({
                "ppt_name": family,
                "path": path,
                "index": index,
                "cjk": True,
                "penalty": 0.0,
                "font_face_id": face.get("family_id"),
                "_catalog_face": dict(face),
            })

        for name in _LATIN_FONT_PREFERENCE:
            path = fontconfig_font_path(f"{name}:style=Regular")
            if not path:
                continue
            try:
                actual_family, _style = ImageFont.truetype(path, 16).getname()
            except OSError:
                continue
            family_key = "".join(ch for ch in actual_family.casefold()
                                 if ch.isalnum())
            marker = (family_key, str(path).casefold(), 0)
            if not actual_family or marker in seen_faces:
                continue
            seen_faces.add(marker)
            out.append({
                "ppt_name": actual_family,
                "path": path,
                "index": 0,
                "cjk": is_cjk_font_family(actual_family),
                "penalty": 0.0,
            })
        # Keep deterministic local-candidate ordering with the fallback
        # face first as a tie-breaker, never as a source of font evidence.
        out.sort(key=lambda item: (
            0 if item["ppt_name"].casefold() == "simhei" else
            0 if item["ppt_name"].casefold() == "times new roman" else 1,
            str(item["ppt_name"]).casefold(),
            str(item["path"]).casefold(),
            int(item.get("index", 0)),
        ))
        for item in out:
            item["penalty"] = 0.0
        _FONT_CANDIDATES_CACHE = out
        candidates = list(out)

    if cjk is not None:
        filtered = [item for item in candidates if item["cjk"] is cjk]
        candidates = filtered
    if text:
        try:
            from layout.font_catalog import face_has_glyph
            usable = []
            codepoints = {
                ord(char) for char in text
                if char.strip() and char not in "\u200b\ufeff"
            }
            for item in candidates:
                face = item.get("_catalog_face")
                if face is not None:
                    if any(face_has_glyph(face, chr(codepoint)) is False
                           for codepoint in codepoints):
                        continue
                usable.append(item)
            candidates = usable
        except Exception:
            pass
    if preferred_family:
        candidates.sort(key=lambda item: (
            0 if font_families_match(preferred_family, item["ppt_name"]) else 1,
            float(item.get("penalty", 0.0)),
        ))
    return candidates


def resolve_ppt_font(family: str | None, *, text: str = "") -> str:
    """Resolve a render-fit family, otherwise use the script fallback."""
    requested = str(family or "").strip()
    fallback = fallback_ppt_font(text)
    if not requested:
        return fallback
    requested_key = "".join(ch for ch in requested.casefold() if ch.isalnum())
    # User retired PuHuiTi from the candidate catalog. Old layout files may
    # still name it, so route those requests through the script fallback.
    if requested_key in {"alibabapuhuiti", "alibabapuhuiti20"}:
        return fallback
    # Keep the explicit fallback names stable even on a machine without the
    # corresponding font file; Office can resolve them on the target machine.
    if requested.casefold() in {"simhei", "times new roman"}:
        return requested
    if requested:
        path = fontconfig_font_path(f"{requested}:style=Regular")
        if path:
            try:
                actual_family, _style = ImageFont.truetype(path, 16).getname()
            except OSError:
                actual_family = ""
            if actual_family and font_families_match(requested, actual_family):
                return actual_family

    candidates = _font_candidates(text=text)
    for candidate in candidates:
        if requested and font_families_match(requested, candidate["ppt_name"]):
            return candidate["ppt_name"]
    return fallback


def _load_font(path: str, pixel_size: int,
               face_index: int = 0) -> ImageFont.FreeTypeFont | None:
    pixel_size = max(1, int(pixel_size))
    face_index = max(0, int(face_index))
    key = (path, pixel_size, face_index)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(
                path, pixel_size, index=face_index)
        except (OSError, ValueError, IndexError):
            return None
    return _FONT_CACHE[key]


def _render_text_metrics(text: str, font_path: str, size_pt: int,
                         pt_per_px: float, bold: bool,
                         face_index: int = 0,
                         italic: bool = False) -> dict | None:
    """Render one text line locally and return its ink bbox in source px.

    The layout stores font size in points, but the source image is in
    pixels. ``pt_per_px`` converts PPT points back to source pixels so
    the rendered mask can be compared directly with the OCR crop.
    """
    if not text or "\n" in text or pt_per_px <= 0:
        return None
    pixel_size = max(1, int(round(float(size_pt) / pt_per_px)))
    face_index = max(0, int(face_index))
    cache_key = (text, font_path, pixel_size, face_index,
                 bool(bold), bool(italic))
    if cache_key in _TEXT_RENDER_CACHE:
        return _TEXT_RENDER_CACHE[cache_key]

    font = _load_font(font_path, pixel_size, face_index)
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

    if italic:
        shear = -0.22
        extra_w = int(round(abs(shear) * canvas_h))
        x_offset = extra_w if shear < 0 else 0
        img = img.transform(
            (canvas_w + extra_w, canvas_h),
            Image.Transform.AFFINE,
            (1.0, shear, float(x_offset), 0.0, 1.0, 0.0),
            resample=Image.Resampling.BICUBIC,
        )
    else:
        x_offset = 0
    arr = np.array(img)
    ys, xs = np.where(arr > 16)
    if ys.size == 0 or xs.size == 0:
        _TEXT_RENDER_CACHE[cache_key] = None
        return None
    x_min = int(xs.min()); x_max = int(xs.max()) + 1
    y_min = int(ys.min()); y_max = int(ys.max()) + 1
    rel_bbox = [
        int(x_min - origin[0] - x_offset), int(y_min - origin[1]),
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

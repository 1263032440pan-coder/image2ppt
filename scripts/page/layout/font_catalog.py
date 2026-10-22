"""Local, non-distributive catalog of the 20 requested font families.

The catalog only indexes font files already installed locally or explicitly
present in configured local directories. It never downloads or copies fonts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from fontconfig_helper import fontconfig_font_path  # noqa: E402


TARGET_FAMILIES: tuple[dict, ...] = (
    {"id": "source_han_sans", "name": "思源黑体", "aliases":
        ("思源黑体", "Source Han Sans", "Source Han Sans SC", "Source Han Sans CN")},
    {"id": "source_han_serif", "name": "思源宋体", "aliases":
        ("思源宋体", "Source Han Serif", "Source Han Serif SC", "Source Han Serif CN")},
    {"id": "oppo_sans", "name": "OPPO Sans", "aliases":
        ("OPPO Sans", "OPPO Sans 4.0")},
    {"id": "harmonyos_sans", "name": "HarmonyOS Sans", "aliases":
        ("HarmonyOS Sans", "HarmonyOS Sans SC", "HarmonyOS Sans SC VF")},
    {"id": "lxgw_wenkai", "name": "霞鹜文楷", "aliases":
        ("霞鹜文楷", "LXGW WenKai", "LXGW WenKai GB")},
    {"id": "smiley_sans", "name": "得意黑", "aliases":
        ("得意黑", "Smiley Sans")},
    {"id": "youshe_biaotihei", "name": "优设标题黑", "aliases":
        ("优设标题黑", "YouShe BiaoTiHei", "YouSheBiaoTiHei")},
    {"id": "pangmen_zhengdao", "name": "庞门正道标题体", "aliases":
        ("庞门正道标题体", "庞门正道标题", "PangMenZhengDao")},
    {"id": "zcool_kuaile", "name": "站酷快乐体", "aliases":
        ("站酷快乐体", "ZCOOL KuaiLe")},
    {"id": "microsoft_yahei", "name": "微软雅黑", "aliases":
        ("微软雅黑", "Microsoft YaHei", "Microsoft YaHei UI")},
    {"id": "simsun", "name": "宋体", "aliases": ("宋体", "SimSun", "NSimSun")},
    {"id": "simhei", "name": "黑体", "aliases": ("黑体", "SimHei")},
    {"id": "kaiti", "name": "楷体", "aliases": ("楷体", "KaiTi", "KaiTi_GB2312")},
    {"id": "fangsong", "name": "仿宋", "aliases": ("仿宋", "FangSong", "FangSong_GB2312")},
    {"id": "dengxian", "name": "等线", "aliases": ("等线", "DengXian")},
    {"id": "pingfang", "name": "苹方", "aliases": ("苹方", "PingFang SC", "PingFang-SC")},
    {"id": "fz_lantinghei", "name": "方正兰亭黑", "aliases":
        ("方正兰亭黑", "FZLanTingHei", "FZLanTingHeiS-R-GB")},
    {"id": "st_zhongsong", "name": "华文中宋", "aliases":
        ("华文中宋", "STZhongsong", "STZhongsong SC")},
    {"id": "lisu", "name": "隶书", "aliases": ("隶书", "LiSu")},
    {"id": "youyuan", "name": "幼圆", "aliases": ("幼圆", "YouYuan")},
)

_SUPPORTED_SUFFIXES = {".ttf", ".otf", ".ttc", ".otc"}
_FONT_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
_FACES: list[dict] | None = None
_HASH_CACHE: dict[str, str] = {}
_CMAP_CACHE: dict[tuple[str, int], set[int] | None] = {}
_CATALOG_ERRORS: list[dict] = []


def _key(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", value.casefold())


def _family_for_name(name: str) -> dict | None:
    candidate = _key(name)
    for family in TARGET_FAMILIES:
        if candidate in {_key(alias) for alias in family["aliases"]}:
            return family
    return None


@lru_cache(maxsize=2048)
def _license_declaration(font_path: Path) -> dict:
    for root in _font_roots():
        try:
            relative = font_path.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        manifest_path = root / "license_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        files = data.get("files", {}) if isinstance(data, dict) else {}
        declaration = files.get(relative.as_posix()) or files.get(font_path.name)
        if isinstance(declaration, dict):
            return declaration
    return {}


def _source_for_path(path: Path) -> tuple[str, bool, str]:
    resolved = str(path.resolve()).casefold()
    system_roots = []
    windir = os.environ.get("WINDIR")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if windir:
        system_roots.append(str((Path(windir) / "Fonts").resolve()).casefold())
    if local_app_data:
        system_roots.append(str((Path(local_app_data) / "Microsoft" / "Windows" / "Fonts").resolve()).casefold())
    if sys.platform == "darwin":
        system_roots.extend(("/system/library/fonts", "/library/fonts", str((Path.home() / "Library/Fonts").resolve()).casefold()))
    else:
        system_roots.extend(("/usr/share/fonts", "/usr/local/share/fonts", str((Path.home() / ".local/share/fonts").resolve()).casefold()))
    if any(resolved.startswith(root.rstrip("\\/") + os.sep.casefold()) or resolved == root
           for root in system_roots if root):
        return "system_installed", True, "system_reference_only"

    declared = _license_declaration(path)
    allowed = bool(declared.get("matching_allowed", False))
    state = "declared_for_local_matching" if allowed else "license_unknown"
    return "configured_local", allowed, state


def _font_roots() -> list[Path]:
    roots = [_FONT_DIR]
    if sys.platform == "darwin":
        roots.extend((Path("/System/Library/Fonts"), Path("/Library/Fonts"),
                      Path.home() / "Library/Fonts"))
    elif os.name != "nt":
        roots.extend((Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
                      Path.home() / ".local/share/fonts"))
    configured = os.environ.get("IMAGE2PPT_FONT_DIRS", "")
    for item in configured.split(os.pathsep):
        if item.strip():
            roots.append(Path(item.strip()).expanduser())
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = str(resolved).casefold()
        if key not in seen:
            seen.add(key)
            out.append(resolved)
    return out


def _enumerate_faces(path: Path) -> list[tuple[int, str, str]]:
    indices = range(64) if path.suffix.casefold() in {".ttc", ".otc"} else range(1)
    faces = []
    for index in indices:
        try:
            font = ImageFont.truetype(str(path), 16, index=index)
            family, style = font.getname()
        except (OSError, IndexError, ValueError):
            break
        if family:
            faces.append((index, str(family), str(style or "")))
    return faces


def _font_paths_and_known_faces() -> tuple[set[Path], list[tuple[str, str, str]]]:
    paths: set[Path] = set()
    known = []
    if os.name == "nt":
        try:
            from fontconfig_helper import _windows_font_records
            for raw_path, family, style in _windows_font_records():
                if _family_for_name(family) is None:
                    continue
                path = Path(raw_path)
                if path.is_file():
                    paths.add(path)
                    known.append((str(path), family, style))
        except (ImportError, OSError):
            pass

    for root in _font_roots():
        if not root.is_dir():
            continue
        try:
            paths.update(p for p in root.rglob("*")
                         if p.is_file() and p.suffix.casefold() in _SUPPORTED_SUFFIXES)
        except OSError:
            continue

    if os.name != "nt":
        for family in TARGET_FAMILIES:
            for alias in family["aliases"]:
                path = fontconfig_font_path(f"{alias}:style=Regular")
                if path:
                    paths.add(Path(path))
    return paths, known


def _sha256(path: Path) -> str:
    key = str(path)
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def _scan_faces() -> list[dict]:
    global _CATALOG_ERRORS
    _CATALOG_ERRORS = []
    paths, known = _font_paths_and_known_faces()
    faces: list[dict] = []
    seen: set[tuple[str, int, str]] = set()
    for path in sorted(paths, key=lambda p: str(p).casefold()):
        try:
            enumerated = _enumerate_faces(path)
        except (OSError, ValueError, IndexError) as exc:
            registered_names = [family for known_path, family, _style in known
                                if str(path) == known_path]
            if any(_family_for_name(name) for name in registered_names):
                _CATALOG_ERRORS.append({
                    "font_file_path": str(path),
                    "status": "parse_error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            continue
        local_source, local_allowed, license_state = _source_for_path(path)
        declaration = (_license_declaration(path)
                       if local_source == "configured_local" else {})
        for index, internal_family, style in enumerated:
            target = _family_for_name(internal_family)
            if target is None:
                continue
            marker = (str(path.resolve()).casefold(), index, target["id"])
            if marker in seen:
                continue
            seen.add(marker)
            try:
                digest = _sha256(path)
            except OSError:
                digest = ""
            cmap = _cmap_for_face(str(path), index)
            try:
                from fontTools.ttLib import TTFont
                parsed_font = TTFont(str(path), fontNumber=index, lazy=True)
                version = parsed_font["name"].getDebugName(5)
                postscript_name = parsed_font["name"].getDebugName(6)
                parsed_font.close()
            except Exception:
                version = None
                postscript_name = None
            cjk_count = (sum(1 for codepoint in cmap
                             if 0x3400 <= codepoint <= 0x9FFF)
                         if cmap is not None else None)
            # A system registry entry confirms local installation; arbitrary
            # configured font files require an explicit local-use declaration.
            is_registered = any(str(path) == known_path and family == internal_family
                                for known_path, family, _style in known)
            matching_allowed = local_allowed or is_registered
            faces.append({
                "font_id": f"{target['id']}:{digest[:16]}:{index}",
                "font_face_id": f"sha256:{digest}:face:{index}" if digest else None,
                "family_id": target["id"],
                "target_family": target["name"],
                "font_family_name": internal_family,
                "postscript_name": postscript_name,
                "style_name": style,
                "font_version": version,
                "font_file_path": str(path.resolve()),
                "font_filename": path.name,
                "face_index": index,
                "file_sha256": digest or None,
                "unicode_cmap_codepoint_count": len(cmap) if cmap is not None else None,
                "cjk_basic_codepoint_count": cjk_count,
                "cmap_status": "readable" if cmap is not None else "unreadable",
                "ppt_font_name": internal_family,
                "source": local_source,
                "license_status": license_state if not is_registered else "system_reference_only",
                "license_id": declaration.get("license_id"),
                "license_source": declaration.get("license_source"),
                "license_reviewed_at": declaration.get("reviewed_at"),
                "local_match_allowed": bool(matching_allowed),
                "redistribution_allowed": (False if is_registered else
                                           declaration.get("redistribution_allowed")),
                "target_font_available": "unknown",
            })
    faces.sort(key=lambda face: (face["family_id"], face["font_family_name"].casefold(),
                                 face["style_name"].casefold(), face["face_index"]))
    return faces


def get_font_faces() -> list[dict]:
    global _FACES
    if _FACES is None:
        _FACES = _scan_faces()
    return [dict(face) for face in _FACES]


def get_font_catalog_manifest() -> dict:
    faces = get_font_faces()
    families = []
    for family in TARGET_FAMILIES:
        family_faces = [face for face in faces if face["family_id"] == family["id"]]
        usable = [face for face in family_faces if face["local_match_allowed"]]
        if not family_faces:
            source_status = "asset_missing"
            license_status = "not_assessed_asset_missing"
        elif not usable:
            source_status = "font_face_found"
            license_status = "license_blocked_or_unknown"
        elif all(face.get("license_status") == "system_reference_only"
                 for face in usable):
            source_status = "system_font_resolved"
            license_status = "system_reference_only"
        elif all(face.get("license_status") == "declared_for_local_matching"
                 for face in usable):
            source_status = "configured_font_resolved"
            license_status = "local_matching_declared"
        else:
            source_status = "font_face_resolved"
            license_status = "mixed_face_license_status"
        families.append({
            "family_id": family["id"],
            "canonical_name": family["name"],
            "aliases": list(family["aliases"]),
            "source_status": source_status,
            "license_status": license_status,
            "availability": ("available" if usable else
                              "license_blocked" if family_faces else "missing_asset"),
            "matching_availability": "available" if usable else (
                "license_blocked" if family_faces else "missing_asset"),
            "face_count": len(family_faces),
            "usable_face_count": len(usable),
            "target_font_availability": "unknown",
            "faces": family_faces,
        })
    return {
        "schema_version": 2,
        "catalog_scope": "20 requested families; local installed/configured files only",
        "catalog_status": "complete" if all(
            family["matching_availability"] == "available"
            for family in families) else "incomplete_with_explicit_blockers",
        "downloads_performed": False,
        "font_embedding_performed": False,
        "target_font_availability_default": "unknown",
        "scan_errors": list(_CATALOG_ERRORS),
        "families": families,
    }


@lru_cache(maxsize=8192)
def _cmap_for_face(path: str, index: int) -> set[int] | None:
    try:
        from fontTools.ttLib import TTFont
        font = TTFont(path, fontNumber=index, lazy=True)
        cmap: set[int] = set()
        for table in font["cmap"].tables:
            if table.isUnicode():
                cmap.update(int(codepoint) for codepoint in table.cmap)
        font.close()
        return cmap
    except Exception:
        return None


def face_has_glyph(face: dict, char: str) -> bool | None:
    cmap = _cmap_for_face(face["font_file_path"], int(face["face_index"]))
    if cmap is None:
        return None
    return ord(char) in cmap

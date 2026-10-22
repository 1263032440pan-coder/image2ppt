#!/usr/bin/env python
"""Fontconfig helpers for local Office fonts used by LibreOffice/Pillow."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from functools import lru_cache
from pathlib import Path


_FONTCONFIG_ENV: dict[str, str] | None = None


_FONT_FAMILY_ALIAS_GROUPS = (
    frozenset({"notosanscjksc", "notosanssc"}),
    frozenset({"notoserifcjksc", "notoserifsc"}),
)


def _family_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", normalized.casefold())


def font_families_match(requested: str, actual: str) -> bool:
    """Return whether two family labels name the same supported font family.

    Some Windows font files expose versioned family names while the project
    catalog may use a shorter alias. Keep supported aliases explicit so the
    PPTX stores a name Office can resolve on the conversion machine.
    """
    a = _family_key(requested)
    b = _family_key(actual)
    if not a or not b:
        return False
    if a == b:
        return True
    return any(a in group and b in group
               for group in _FONT_FAMILY_ALIAS_GROUPS)


def _split_font_query(query: str) -> tuple[str, str | None]:
    match = re.match(r"^(.*?)(?::style=(.*))?$", str(query or "").strip())
    if not match:
        return str(query or "").strip(), None
    return match.group(1).strip(), (match.group(2) or "").strip() or None


@lru_cache(maxsize=1)
def _windows_font_records() -> tuple[tuple[str, str, str], ...]:
    """Read installed Windows font files and their internal family names.

    File names and registry display names are not reliable family names.
    Pillow/FreeType reads the name table Office uses, which also lets us map
    aliases without guessing from filenames.
    """
    if os.name != "nt":
        return ()
    try:
        import winreg
        from PIL import ImageFont
    except ImportError:
        return ()

    roots = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        roots.append(Path(local_app_data) / "Microsoft" / "Windows" / "Fonts")
    raw_paths: set[Path] = set()
    registry_keys = (
        (winreg.HKEY_LOCAL_MACHINE,
         r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
        (winreg.HKEY_CURRENT_USER,
         r"Software\Microsoft\Windows NT\CurrentVersion\Fonts"),
    )
    for hive, key_path in registry_keys:
        try:
            with winreg.OpenKey(hive, key_path) as key:
                value_count = winreg.QueryInfoKey(key)[1]
                for index in range(value_count):
                    _label, value, _kind = winreg.EnumValue(key, index)
                    if not isinstance(value, str):
                        continue
                    font_path = Path(os.path.expandvars(value))
                    if not font_path.is_absolute():
                        font_path = roots[0] / font_path
                    raw_paths.add(font_path)
        except OSError:
            continue

    # Include standard font directories as a fallback for per-user installs
    # whose registry entry only stores a file name.
    for root in roots:
        if root.is_dir():
            try:
                raw_paths.update(root.iterdir())
            except OSError:
                continue

    records: list[tuple[str, str, str]] = []
    supported = {".ttf", ".otf", ".ttc", ".otc"}
    for path in sorted(raw_paths, key=lambda p: str(p).casefold()):
        if path.suffix.casefold() not in supported or not path.is_file():
            continue
        indices = range(8) if path.suffix.casefold() in {".ttc", ".otc"} else range(1)
        for index in indices:
            try:
                face = ImageFont.truetype(str(path), 16, index=index)
                family, style = face.getname()
            except (OSError, IndexError, ValueError):
                if index == 0:
                    continue
                break
            if family:
                records.append((str(path), str(family), str(style or "")))
    return tuple(records)


def _style_rank(style: str, requested: str | None) -> tuple[int, int]:
    value = _family_key(style)
    wanted = _family_key(requested or "regular")
    is_bold = "bold" in value or "black" in value or "demi" in value
    is_italic = "italic" in value or "oblique" in value
    wants_bold = "bold" in wanted
    wants_italic = "italic" in wanted or "oblique" in wanted
    mismatch = int(is_bold != wants_bold) + int(is_italic != wants_italic)
    regular_rank = 0 if any(x in value for x in ("regular", "normal", "book")) else 1
    if wants_bold and is_bold:
        regular_rank = 0
    return mismatch, regular_rank


@lru_cache(maxsize=256)
def _windows_font_path(family: str, style: str | None) -> str | None:
    matches = [record for record in _windows_font_records()
               if font_families_match(family, record[1])]
    if not matches:
        return None
    matches.sort(key=lambda record: _style_rank(record[2], style))
    return matches[0][0]


def _office_yahei_dirs() -> list[Path]:
    roots = [
        Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts"),
        Path("/Applications/Microsoft Word.app/Contents/Resources/DFonts"),
        Path("/Applications/Microsoft Excel.app/Contents/Resources/DFonts"),
        Path("/Applications/Microsoft OneNote.app/Contents/Resources/DFonts"),
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        if root in seen:
            continue
        if (root / "msyh.ttc").exists():
            out.append(root)
            seen.add(root)
    return out


def fontconfig_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return an env where Office's YaHei TTCs are visible to fontconfig."""
    global _FONTCONFIG_ENV
    env = dict(base or os.environ)
    dirs = _office_yahei_dirs()
    if not dirs:
        return env
    if _FONTCONFIG_ENV is None:
        conf_path = Path(tempfile.gettempdir()) / "image2ppt-fonts.conf"
        dir_xml = "\n".join(f"  <dir>{d}</dir>" for d in dirs)
        conf_path.write_text(
            "<?xml version=\"1.0\"?>\n"
            "<!DOCTYPE fontconfig SYSTEM \"fonts.dtd\">\n"
            "<fontconfig>\n"
            "  <include ignore_missing=\"yes\">"
            "/opt/homebrew/etc/fonts/fonts.conf</include>\n"
            f"{dir_xml}\n"
            "</fontconfig>\n",
            encoding="utf-8",
        )
        _FONTCONFIG_ENV = {"FONTCONFIG_FILE": str(conf_path)}
    env.update(_FONTCONFIG_ENV)
    return env


def fontconfig_font_path(query: str) -> str | None:
    """Find a real local font file for a requested family/style.

    On macOS/Linux, fontconfig is preferred. Windows does not normally ship
    ``fc-match``, so use the installed-font registry and the internal family
    names exposed by FreeType. A fontconfig fallback to an unrelated family
    is rejected; callers must know which family their metrics came from.
    """
    family, style = _split_font_query(query)
    if os.name == "nt":
        return _windows_font_path(family, style)

    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}", query],
            check=True,
            capture_output=True,
            text=True,
            env=fontconfig_env(),
        )
        path = Path(result.stdout.strip())
        if path.is_file():
            try:
                from PIL import ImageFont
                actual_family, _actual_style = ImageFont.truetype(
                    str(path), 16).getname()
            except (ImportError, OSError, ValueError):
                actual_family = ""
            if actual_family and font_families_match(family, actual_family):
                return str(path)
    except (OSError, subprocess.CalledProcessError):
        pass

    return None

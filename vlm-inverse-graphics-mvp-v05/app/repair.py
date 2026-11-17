"""Local colour refinement without SLIC polygons or visible cell seams."""
from __future__ import annotations
from .color import refine_spatial_band


def refine_worst_band(image_bytes: bytes, color_ir: dict, region_report: dict,
                      max_extra: int = 1100) -> tuple[dict, dict]:
    """Add measured gradient stops in the worst-scoring spatial band only.

    This intentionally never adds raster tiles or SVG coloured polygons.
    """
    bands = {'sky': (0, .31), 'midground': (.31, .52), 'foreground_water': (.52, 1)}
    regions = [r for r in region_report.get('regions', []) if r.get('id') in bands]
    if color_ir.get('mode') != 'continuous_scanline_gradients' or not regions:
        return color_ir, {'modified': False, 'reason': 'No continuous field or scored region'}
    worst = min(regions, key=lambda r: r['color_score'])
    name = worst['id']; lo, hi = bands[name]
    # Full-width gradient strokes are editable, finer horizontal sampling only
    # replaces the target bands; unrelated rows are byte-for-byte unchanged.
    current = int(color_ir.get('statistics', {}).get('sample_step', 10))
    if current <= 3:
        return color_ir, {'modified': False, 'region': name, 'reason': 'At maximum stop density'}
    finer = max(3, current // 2)
    refined, changed = refine_spatial_band(image_bytes, color_ir,
                                           round(lo * 1000), round(hi * 1000), finer)
    return refined, {'modified': changed > 0, 'region': name, 'refined_bands': changed,
                     'sample_step': finer, 'baseline_color_score': worst['color_score']}

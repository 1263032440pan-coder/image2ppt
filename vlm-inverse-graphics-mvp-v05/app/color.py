"""v0.5: edge-preserving high-resolution continuous SVG colour field.

Colour samples come directly from the uploaded image. A one-pixel SVG scanline has its own horizontal linearGradient. Original
image details are sampled at output resolution, without the 640px downsample
or Gaussian blurring previously responsible for the visibly soft image.

This is high-fidelity image-guided vector approximation, *not* semantic shape
synthesis. Geometry IR and Detail IR remain separately editable.
"""
from __future__ import annotations

import io
import os

import cv2
import numpy as np
from PIL import Image


def _hex(rgb) -> str:
    return '#' + ''.join(f'{int(np.clip(x, 0, 255)):02X}' for x in rgb)


def _dominant_palette(rgb: np.ndarray, colors: int = 12) -> list[dict]:
    sample = Image.fromarray(rgb, 'RGB').resize((256, 256))
    quantized = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    counts = sorted(quantized.getcolors() or [], reverse=True)
    palette = quantized.getpalette() or []
    total = max(1, sum(n for n, _ in counts))
    return [{'color': _hex(palette[i * 3: i * 3 + 3]), 'fraction': round(n / total, 4)}
            for n, i in counts[:colors]]


def _bounded_env(name: str, default: int, low: int, high: int) -> int:
    try:
        val = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        val = default
    return max(low, min(high, val))


def _color_field(image_bytes: bytes, size: int = 1000) -> np.ndarray:
    """Preserve source edges: no blur, no enlargement from a 640px thumbnail.

    A single resize to canvas resolution is unavoidable. INTER_AREA reduces
    aliasing when reducing a large image; INTER_CUBIC is used when enlarging.
    Semantic vector contours and detail strokes remain independent of the
    source-measured colour field.
    """
    raw = np.asarray(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    if raw.shape[1] == size and raw.shape[0] == size:
        return raw.copy()
    method = cv2.INTER_AREA if min(raw.shape[:2]) >= size else cv2.INTER_CUBIC
    return cv2.resize(raw, (size, size), interpolation=method)


def _bands_from_field(field: np.ndarray, y0: int, y1: int, y_step: int,
                      x_step: int) -> list[dict]:
    """Sample source colour at high resolution and preserve sharp transitions.

    Regular sparse stops are sufficient for smooth gradients. Strong horizontal
    transitions receive adjacent stops so a narrow ink stroke does not become
    a smeared multi-pixel interpolation. No visible cell edges are drawn.
    """
    height, width = field.shape[:2]
    baseline_x = list(range(0, width - 1, x_step)) + [width - 1]
    bands = []
    for y in range(y0, min(y1, height), y_step):
        next_y = min(y + y_step, height)
        # Normal operation uses 1px strips: never average neighbouring source
        # rows, which softened thin horizontal ink strokes in the prior build.
        row = field[y] if next_y == y + 1 else field[y:next_y].mean(axis=0)
        slope = np.max(np.abs(np.diff(row.astype(np.int16), axis=0)), axis=1)
        candidates = np.flatnonzero(slope > 17)
        # Bound SVG complexity on highly textured inputs. Prioritise the
        # strongest edges, then preserve BOTH ends of each chosen transition.
        if len(candidates) > 125:
            candidates = candidates[np.argsort(slope[candidates])[-125:]]
        xs = sorted(set(baseline_x)
                    .union(int(x) for x in candidates)
                    .union(int(x + 1) for x in candidates))
        stops = [[round(x / (width - 1) * 100, 4), _hex(row[x])]
                 for x in xs]
        bands.append({'y': y, 'height': next_y - y, 'stops': stops})
    return bands


def reconstruct_colors(image_bytes: bytes, scene: dict, geometry: dict | None = None,
                       enabled: bool = True, target_segments: int | None = None,
                       band_height: int | None = None, sample_step: int | None = None) -> dict:
    """Build an SVG-ready continuous spatial Color IR.

    `target_segments` is accepted for older pipeline/test callers but is no
    longer used: cell tessellation was the source of unwanted grid artefacts.
    """
    if not enabled:
        return {'version': '0.5', 'mode': 'disabled', 'bands': [], 'palette': [], 'statistics': {}}
    y_step = max(1, min(4, band_height if band_height is not None else _bounded_env('COLOR_BAND_HEIGHT', 1, 1, 4)))
    x_step = max(3, min(25, sample_step if sample_step is not None else _bounded_env('COLOR_SAMPLE_STEP', 6, 3, 25)))
    img = _color_field(image_bytes)
    bands = _bands_from_field(img, 0, img.shape[0], y_step, x_step)
    return {
        'version': '0.5', 'mode': 'continuous_scanline_gradients',
        'background': _hex(np.rint(img.reshape(-1, 3).mean(axis=0))),
        'palette': _dominant_palette(img),
        'bands': bands,
        'statistics': {
            'bands': len(bands), 'color_stops': sum(len(b['stops']) for b in bands),
            'band_height': y_step, 'sample_step': x_step,
            'method': 'source-resolution unblurred colour field, edge-adaptive single-pixel SVG gradient scanlines',
            'source_bitmap_embedded': False, 'slic_cells_rendered': False,
            'grid_borders': False,
            'target_segments_deprecated': target_segments is not None,
        },
    }


def refine_spatial_band(image_bytes: bytes, color_ir: dict, y0: int, y1: int,
                        sample_step: int = 5) -> tuple[dict, int]:
    """Increase stop density in one band without reintroducing polygon patches."""
    if color_ir.get('mode') != 'continuous_scanline_gradients':
        return color_ir, 0
    y0, y1 = max(0, min(1000, y0)), max(0, min(1000, y1))
    if y1 <= y0:
        return color_ir, 0
    source = _color_field(image_bytes)
    updated = []
    count = 0
    for band in color_ir.get('bands', []):
        y = int(band['y'])
        if y >= y0 and y < y1:
            rebuilt = _bands_from_field(source, y, y + int(band['height']), int(band['height']), sample_step)
            if rebuilt:
                updated.append(rebuilt[0]); count += 1
                continue
        updated.append(band)
    refined = {**color_ir, 'bands': updated,
               'statistics': {**color_ir.get('statistics', {}),
                              'color_stops': sum(len(b['stops']) for b in updated),
                              'last_refined_rows': count}}
    return refined, count

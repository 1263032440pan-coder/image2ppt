"""Detail IR v0.5: sharp source-grounded fine strokes for the whole composition.

A contour is an image-derived polyline, not model-generated SVG. This restores
line work outside the koi (bridge rails, house grids, pine needles, ripples).
It cannot recover true semantic parts; see geometry parts for those.
"""
from __future__ import annotations
import io
import os

import cv2
import numpy as np
from PIL import Image


def reconstruct_details(image_bytes: bytes, scene: dict,
                        enabled: bool = True, max_paths: int | None = None) -> dict:
    if not enabled:
        return {'version': '0.5', 'mode': 'disabled', 'strokes': [], 'statistics': {}}
    image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((1000,1000), resample=Image.Resampling.LANCZOS))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    smooth = gray  # do not blur narrow source ink or foliage needles
    edges = cv2.Canny(smooth, 55, 125)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    limit = max(200, min(7000, max_paths or int(os.getenv('DETAIL_MAX_PATHS', '4800'))))
    contours = sorted((c for c in contours if cv2.arcLength(c, False) >= 8),
                      key=lambda c: cv2.arcLength(c, False), reverse=True)[:limit]
    strokes = []
    for contour in contours:
        contour = cv2.approxPolyDP(contour, 0.45, False).reshape(-1,2)
        if len(contour) < 2:
            continue
        if len(contour) > 450:
            contour = contour[::int(np.ceil(len(contour) / 450))]
        mx,my = map(int, contour[len(contour)//2])
        # Ink color is sampled from the darkest nearby pixels rather than the
        # fixed brown/black ink that made previous iterations look inaccurate.
        x0,x1=max(0,mx-2),min(1000,mx+3)
        y0,y1=max(0,my-2),min(1000,my+3)
        tile=image[y0:y1,x0:x1].reshape(-1,3)
        ink=tile[np.argmin(tile.mean(axis=1))].astype(int)
        ink=np.clip(ink * .88, 0, 255).astype(int)
        color='#'+''.join(f'{int(x):02X}' for x in ink)
        # Long spurious chords are introduced when aggressively decimating a
        # folded Canny contour. They are NOT real image lines: split rather
        # than connecting two remote samples with a diagonal SVG segment.
        parts=[]; current=[contour[0]]
        for pt in contour[1:]:
            if np.linalg.norm(pt.astype(np.float32)-current[-1].astype(np.float32)) > 13:
                if len(current)>=2:
                    parts.append(current)
                current=[pt]
            else:
                current.append(pt)
        if len(current)>=2:
            parts.append(current)
        for segment in parts:
            points=[[round(float(px),2),round(float(py),2)] for px,py in segment]
            strokes.append({'points': points, 'color': color, 'width': 0.85,
                            'opacity': 0.85})
    return {'version': '0.5', 'mode': 'source_edge_paths', 'strokes': strokes,
            'statistics': {'path_count':len(strokes), 'source_bitmap_embedded':False,
                           'algorithm':'Canny contours with long-jump splitting'}}

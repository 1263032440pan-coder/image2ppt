"""Regression: ink must remain sharp without bringing back polygon grids."""
import io
import os
from PIL import Image
import numpy as np
from app.color import _color_field, reconstruct_colors
from app.detail import reconstruct_details
from app.graphics import scene_to_svg
from app.schema import normalize_scene


def test_one_pixel_horizontal_ink_is_not_blurred_away():
    rgb = np.full((1000, 1000, 3), 220, dtype=np.uint8)
    rgb[500, 100:900] = [15, 18, 19]
    b = io.BytesIO(); Image.fromarray(rgb).save(b, format='PNG')
    measured = _color_field(b.getvalue())
    assert np.array_equal(measured[500, 500], rgb[500, 500])
    assert np.array_equal(measured[499, 500], rgb[499, 500])
    colors = reconstruct_colors(b.getvalue(), {}, enabled=True)
    assert colors['statistics']['band_height'] == 1
    assert colors['bands'][500]['y'] == 500
    assert all(b['height'] == 1 for b in colors['bands'])


def test_detail_overlay_is_configurable_and_conservative(monkeypatch):
    scene = normalize_scene({'objects': [{'id':'area','type':'rock','bbox':[0,0,1000,1000]}]})
    image = np.full((96, 96, 3), 220, dtype=np.uint8)
    image[44:48, 4:90] = 10
    b=io.BytesIO(); Image.fromarray(image).save(b, format='PNG')
    col = reconstruct_colors(b.getvalue(), scene)
    det = reconstruct_details(b.getvalue(), scene)
    monkeypatch.delenv('DETAIL_OVERLAY_STRENGTH', raising=False)
    svg = scene_to_svg(scene, {'objects': {}}, col, det)
    assert 'data-overlay-strength="0.20"' in svg
    assert 'data-mode="continuous-gradient-field"' in svg
    assert '<image' not in svg
    monkeypatch.setenv('DETAIL_OVERLAY_STRENGTH', '0')
    svg = scene_to_svg(scene, {'objects': {}}, col, det)
    assert 'data-overlay-strength="0.00"' in svg
    monkeypatch.setenv('DETAIL_OVERLAY_STRENGTH', '1.5')
    svg = scene_to_svg(scene, {'objects': {}}, col, det)
    assert 'data-overlay-strength="1.00"' in svg

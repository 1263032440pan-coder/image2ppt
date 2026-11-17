"""v0.4 measured continuous colour, geometry/detail and refinement invariants."""
import io
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
from app.schema import normalize_scene
from app.color import reconstruct_colors
from app.detail import reconstruct_details
from app.graphics import scene_to_svg
from app.region_metrics import region_comparison
from app.segmentation import visible_instance_masks
from app.repair import refine_worst_band


def _scene():
    return normalize_scene({'objects': [{'id': 'left', 'type': 'rock',
                                        'bbox': [0, 0, 500, 1000], 'layer': 1}]})


def _checker():
    rgb = np.zeros((96, 96, 3), dtype='uint8')
    rgb[:, :48] = [215, 85, 50]
    rgb[:, 48:] = [55, 122, 202]
    out = io.BytesIO(); Image.fromarray(rgb).save(out, format='PNG')
    return out.getvalue()


def test_continuous_color_spatially_located_and_safe():
    src = _checker(); scene = _scene()
    ir = reconstruct_colors(src, scene, enabled=True, target_segments=750)
    assert ir['mode'] == 'continuous_scanline_gradients'
    assert ir['statistics']['bands'] >= 250
    assert ir['statistics']['grid_borders'] is False
    assert 'patches' not in ir
    assert not ir['statistics']['source_bitmap_embedded']
    band = ir['bands'][100]
    left = [c for x, c in band['stops'] if x < 40]
    right = [c for x, c in band['stops'] if x > 60]
    assert left and right
    assert np.mean([int(c[1:3], 16) for c in left]) > 170
    assert np.mean([int(c[5:7], 16) for c in right]) > 140
    details = reconstruct_details(src, scene, max_paths=400)
    svg = scene_to_svg(scene, {'objects': {}}, ir, details)
    ET.fromstring(svg)
    assert '<image' not in svg
    assert 'data-mode="continuous-gradient-field"' in svg
    assert '<linearGradient ' in svg
    assert 'data-object-id="left"' in svg
    assert 'data-region-owner' not in svg
    assert 'stroke-width="1.3"' not in svg


def test_region_metric_source_alignment_and_heatmap(tmp_path):
    src = _checker(); scene = _scene()
    r = region_comparison(src, src, scene, heatmap_path=tmp_path/'delta.png')
    assert r['global']['mean_delta_e00'] < .01
    assert r['global']['color_score'] > .999
    assert (tmp_path/'delta.png').exists()
    shifted = Image.open(io.BytesIO(src))
    shifted = Image.fromarray(np.asarray(shifted)[:, ::-1].copy())
    b = io.BytesIO(); shifted.save(b, format='PNG')
    wrong = region_comparison(src, b.getvalue(), scene)
    assert wrong['global']['mean_delta_e00'] > 20


def test_refinement_locality_and_instance_occlusion():
    src = _checker(); scene = _scene()
    colors = reconstruct_colors(src, scene, sample_step=12)
    report = {'regions': [{'id': 'sky', 'color_score': .9},
                          {'id': 'midground', 'color_score': .6},
                          {'id': 'foreground_water', 'color_score': .95}]}
    refined, note = refine_worst_band(src, colors, report)
    assert note['modified'] and note['region'] == 'midground'
    assert len(refined['bands']) == len(colors['bands'])
    for before, after in zip(colors['bands'], refined['bands']):
        if .31 * 1000 <= before['y'] < .52 * 1000:
            assert len(after['stops']) >= len(before['stops'])
        else:
            assert after == before
    assert 'patches' not in refined
    scene2 = normalize_scene({'objects': [
        {'id': 'back', 'type': 'koi', 'bbox': [0, 0, 1000, 1000], 'layer': 1},
        {'id': 'front', 'type': 'koi', 'bbox': [0, 0, 1000, 1000], 'layer': 2}]})
    sq = [[150, 150], [850, 150], [850, 850], [150, 850]]
    geo = {'objects': {'front': {'outline': [sq]}, 'back': {'outline': [sq]}}}
    masks = visible_instance_masks(scene2, geo, 256)
    assert masks['front'].sum() > 0
    assert masks['back'].sum() == 0


def test_no_grid_in_smooth_gradient(tmp_path):
    """Gradient-only output must not show false outlines at band boundaries."""
    import cairosvg
    w, h = 128, 128
    yy, xx = np.mgrid[0:h, 0:w]
    src_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    src_rgb[:, :, 0] = np.rint(170 + 30 * yy / h + 10 * xx / w).astype('uint8')
    src_rgb[:, :, 1] = np.rint(190 + 25 * xx / w).astype('uint8')
    src_rgb[:, :, 2] = 170
    src_file = io.BytesIO(); Image.fromarray(src_rgb).save(src_file, format='PNG')
    scene = _scene()
    color = reconstruct_colors(src_file.getvalue(), scene)
    svg = scene_to_svg(scene, {'objects': {}}, color, None)
    png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    rendered = np.asarray(Image.open(io.BytesIO(png)).convert('RGB')).astype('int16')
    # No sharp strip boundaries in the smooth region; a 2-pixel strip height
    # can yield subpixel rounding, but must not create visible bands.
    row_jump = np.max(np.abs(np.diff(rendered[30:970, 50:950], axis=0)), axis=2)
    assert np.percentile(row_jump, 99.9) < 8
    assert (row_jump > 14).mean() < .0001


def test_detail_paths_do_not_join_unrelated_points():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / 'examples/koi_ukiyoe.jpg').read_bytes()
    data = reconstruct_details(source, _scene(), max_paths=120)
    for stroke in data['strokes']:
        pts = np.array(stroke['points'], dtype=np.float32)
        if len(pts) > 1:
            jumps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            assert jumps.max() <= 16.26

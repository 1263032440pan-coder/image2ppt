import json
from pathlib import Path
from xml.etree import ElementTree as ET
from PIL import Image
from io import BytesIO
from app.geometry import extract_geometry
from app.graphics import scene_to_svg
from app.schema import normalize_scene
from app.metrics import compare_objects
from app.pipeline import ROOT, screenshot


def test_source_grounded_paths_and_local_scores(tmp_path):
    scene=normalize_scene(json.loads((ROOT/'examples/demo_scene.json').read_text()))
    original=(ROOT/'examples/koi_ukiyoe.jpg').read_bytes()
    geometry=extract_geometry(original,scene,enabled=True)
    assert {'koi_gold','koi_red','koi_black'}.issubset(geometry['objects'])
    for koi in ('koi_gold','koi_red','koi_black'):
        g=geometry['objects'][koi]
        assert g['mode']=='traced'
        assert len(g['outline'][0])>=10
        assert len(g['regions'])>=5
    svg=scene_to_svg(scene,geometry)
    ET.fromstring(svg)
    assert 'data-object-id="koi_red" data-render-mode="traced"' in svg
    assert '<image' not in svg
    assert ' C' in svg  # real Catmull-Rom-derived cubic Bezier paths
    svg_path=tmp_path/'trace.svg';png_path=tmp_path/'trace.png'
    svg_path.write_text(svg)
    screenshot(svg_path,png_path)
    scores=compare_objects(original,png_path.read_bytes(),scene,geometry)
    assert any(x['id']=='koi_red' and x['render_mode']=='traced' for x in scores)
    assert all(0<=x['score']<=1 for x in scores)


def test_background_and_path_injection_blocked():
    scene=normalize_scene({'objects':[{'id':'koi', 'type':'koi',
        'geometry_hint':{'mode':'<script>', 'foreground_points':[['bad','evil'],[450,650],[float('nan'),12]],
                         'background_points':[[1500,-20]]},
        'bbox':[100,200,400,300]}]})
    assert scene['objects'][0]['geometry_hint']['mode']=='auto'
    assert scene['objects'][0]['geometry_hint']['foreground_points']==[[450,650]]
    assert scene['objects'][0]['geometry_hint']['background_points']==[[1000,0]]
    assert extract_geometry(b'not-an-image',scene,enabled=False)['objects']=={}
    assert 'data-render-mode="procedural"' in scene_to_svg(scene)

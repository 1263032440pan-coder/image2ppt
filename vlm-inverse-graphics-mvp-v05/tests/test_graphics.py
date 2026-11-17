import json
from pathlib import Path
from xml.etree import ElementTree as ET
from app.graphics import scene_to_svg
from app.pipeline import screenshot, ROOT


def test_sample_scene_is_renderable(tmp_path):
    scene=json.loads((ROOT/'examples/demo_scene.json').read_text(encoding='utf-8'))
    svg=scene_to_svg(scene)
    ET.fromstring(svg)
    assert 'data-object-id="koi_red"' in svg
    assert svg.count('data-object-id=')==len(scene['objects'])
    svg_path=tmp_path/'test.svg';png_path=tmp_path/'test.png'
    svg_path.write_text(svg)
    renderer=screenshot(svg_path,png_path)
    assert renderer in ('playwright','cairosvg_fallback')
    assert png_path.stat().st_size>1000

"""Observation -> understanding -> design -> procedural graphics -> review loop."""
from __future__ import annotations
import io
import json
import os
from pathlib import Path
from typing import Callable
from PIL import Image
from .graphics import scene_to_svg
from .geometry import extract_geometry
from .color import reconstruct_colors
from .detail import reconstruct_details
from .region_metrics import region_comparison
from .repair import refine_worst_band
from .metrics import compare_images, compare_objects
from .model import MessagesClient, ModelAPIError
from .schema import KINDS, apply_patches, normalize_scene
from .upload import to_jpeg_bytes

ROOT=Path(__file__).resolve().parents[1]

OBS_PROMPT="""你是二维逆向图形学的 Observation Agent。请仔细观察上传的图像。
目标不是重新生成图片，而是精确理解可程序化重建的场景结构。
使用 1000×1000 归一化坐标，bbox 表示 [x,y,width,height]；图层值越大越靠前。
只输出一个 JSON 对象，禁止 Markdown。至少包含：
{"summary":"...","style":{"medium":"...","outline":"...","colors":["#RRGGBB"]},
 "composition":"...","objects":[{"id":"koi_red","type":"koi","bbox":[500,600,400,200],
 "appearance":"...","relation":"...","layer":12,"fill":"#CC5533"}],
 "constraints":["..."],"uncertainty":["..."]}
识别对象及遮挡：太阳、山、池塘、桥、房屋、松树、锦鲤、荷花、荷叶、芦苇、岩石和水纹。
请着重估计主体准确坐标，不要捏造看不到的对象。对复杂对象可选填
geometry_hint: {"mode":"trace","foreground_points":[[x,y]],"background_points":[[x,y]]}，
其中点必须是图中对应对象内部/外部的可靠位置。重叠的两条鱼务必给出不同对象的内部点。
没有把握就不要输出点坐标，不要输出自由 SVG Path。
"""

DESIGN_PROMPT="""你是 Visual Graphics Design Agent。将观察结果转换成严格的 Scene IR，
用于确定性的 SVG 参数化渲染器。只返回一个 JSON 对象，不要 Markdown。
输出格式：{"background":"#EFE3C8","objects":[
 {"id":"unique_id","type":"koi","bbox":[450,600,370,200],"layer":10,
 "fill":"#D35B45","stroke":"#293535","params":{"rotation":-12,"pattern":"red_white","facing":"left","seed":5}}]}
画布归一化为 1000×1000；bbox = [x,y,width,height]，layer 值大的后画。
仅允许 type 为：sun,mountain,pond,bridge,house,pine,koi,lotus,lily_pad,reed,rock,ripple,bush,cloud。
可用 params: rotation(-180..180), density(1..30), seed(int), arch(0.05..0.85), facing(left/right), pattern(red_white/gold/black/white)。
确保整图主要对象都出现，保留远近遮挡关系。对象总数建议 12-30。
将 Observation 中可靠的 geometry_hint 保留到对应 id 的对象：
"geometry_hint":{"mode":"trace","foreground_points":[[x,y]],"background_points":[[x,y]]}。
以原图中具体对象的轮廓为准：不要让所有锦鲤共享同一个固定鱼形模板。
颜色必须为六位十六进制。禁止输出代码、SVG 路径或多余字段。
以下是观察结果：\n"""

REVIEW_PROMPT="""你是负责几何校对的视觉 Review Agent。
第一张图片是原图，第二张图片是 SVG 程序渲染图。
优先查看下面的 object_metrics 中分数最低的主体，检查具体轮廓、尺寸、颜色、数量、遮挡。
如果一个对象已经由源图 trace，修改 bbox 会重新计算轮廓；也可以调整 geometry_hint 的
foreground_points / background_points（1000 归一化坐标）。
不要仅根据总体分数判断对象像不像。
只能对已存在的对象做局部修复。返回 JSON：
{"summary":"...", "issues":["..."],"patches":[
 {"id":"koi_red","bbox":[500,600,430,200],"fill":"#D45A3C","params":{"rotation":-10}}]}
patches 最多 5 项。仅使用场景中存在的 id。没有可靠的修改建议则返回空数组。
完全禁止输出 JavaScript、SVG、HTML 或自由代码。当前 Scene IR：\n"""


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def screenshot(svg_path: Path, png_path: Path) -> str:
    """Run SVG rasterization in an isolated bounded process, no inherited pipes.

    Some Chromium subprocesses hold their parent's stdout/stderr open; capturing
    pipes can make subprocess.run hang even after a timeout. Use an explicit
    status sidecar file and redirect output to DEVNULL instead.
    """
    import subprocess
    import sys
    status_path=png_path.with_suffix('.renderer.txt')
    status_path.unlink(missing_ok=True)
    try:
        task=subprocess.run([sys.executable,'-m','app.render_worker',
                             str(svg_path),str(png_path),str(status_path)],
                            cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                            timeout=65,check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('SVG render worker timed out after 65 seconds') from exc
    if task.returncode!=0 or not png_path.is_file():
        raise RuntimeError(f'SVG render worker failed with exit code {task.returncode}')
    result=status_path.read_text().strip() if status_path.is_file() else 'isolated_renderer'
    status_path.unlink(missing_ok=True)
    return result


def observe(client: MessagesClient, image: bytes) -> dict:
    result=client.json(OBS_PROMPT,[image],max_tokens=5000)
    if not isinstance(result.get('objects'),list):
        raise ModelAPIError('Observation response must contain objects array.')
    return result


def design(client: MessagesClient, observation: dict) -> dict:
    raw=client.json(DESIGN_PROMPT+json.dumps(observation,ensure_ascii=False),max_tokens=6500)
    scene=normalize_scene(raw.get('scene',raw))
    hints={o.get('id'):o.get('geometry_hint') for o in observation.get('objects', [])
           if isinstance(o,dict) and isinstance(o.get('geometry_hint'),dict)}
    # Recover source-grounded object markers if Design omitted them.
    for obj in scene['objects']:
        if obj['id'] in hints and not obj['geometry_hint']['foreground_points']:
            proposed={**obj,'geometry_hint':hints[obj['id']]}
            from .schema import normalize_object
            validated=normalize_object(proposed)
            if validated:
                obj['geometry_hint']=validated['geometry_hint']
    return scene


def review(client: MessagesClient, original: bytes, rendered: bytes, scene: dict, metrics: dict) -> dict:
    prompt=REVIEW_PROMPT + json.dumps({"scene":scene,"metrics":metrics},ensure_ascii=False)
    return client.json(prompt,[original,rendered],max_tokens=3500)


def run_pipeline(job_dir: Path, image: bytes, iterations: int, demo: bool,
                 progress: Callable[[str,int,str],None], geometry_enabled: bool = True,
                 color_enabled: bool = True, detail_enabled: bool = True) -> dict:
    """v0.5: observation -> geometry -> sharp continuous color -> detail -> visual review.

    Demo uses known sample Scene IR, *not* live image observation, and only one
    rendering pass. When enabled, true review requires an API key and images.
    """
    job_dir.mkdir(parents=True,exist_ok=True)
    (job_dir/'original.jpg').write_bytes(image)
    client=None if demo else MessagesClient()
    progress('observing',10,'观察对象、构图与前后景')
    if demo:
        scene=normalize_scene(json.loads((ROOT/'examples/demo_scene.json').read_text(encoding='utf-8')))
        observation={'summary':'Fixed demonstration scene; not real model observation','mode':'demo',
                     'objects':[{'id':o['id'],'type':o['type'],'bbox':o['bbox']} for o in scene['objects']]}
    else:
        observation=observe(client,image)
    dump(job_dir/'observation.json',observation)
    progress('designing',24,'生成结构化 Scene IR')
    if not demo:
        scene=design(client,observation)
    dump(job_dir/'scene.json',scene)
    iterations=max(1,min(int(os.getenv('MAX_ITERATIONS','3')),int(iterations)))
    history=[]
    best=None
    current=scene
    color_ir=None
    details_ir=None
    refine_info=None
    for i in range(iterations):
        pct=27+int(i*58/max(1,iterations))
        progress('extracting',pct,f'第 {i+1}/{iterations} 轮：实例 Mask 与 Geometry IR')
        geometry=extract_geometry(image,current,enabled=geometry_enabled)
        dump(job_dir/f'iter_{i+1}_geometry.json',geometry)
        if color_ir is None:
            progress('coloring',pct+7,'从原图构建连续颜色场（无网格）')
            color_ir=reconstruct_colors(image,current,geometry,enabled=color_enabled)
        if details_ir is None:
            progress('detailing',pct+10,'提取桥梁、房屋、植被和水纹的细节路径')
            details_ir=reconstruct_details(image,current,enabled=detail_enabled)
        dump(job_dir/f'iter_{i+1}_color.json',color_ir)
        dump(job_dir/f'iter_{i+1}_detail.json',details_ir)
        progress('rendering',pct+15,'生成可编辑 SVG，并渲染浏览器截图')
        svg=scene_to_svg(current,geometry,color_ir,details_ir)
        svg_path=job_dir/f'iter_{i+1}.svg'
        png_path=job_dir/f'iter_{i+1}.png'
        svg_path.write_text(svg,encoding='utf-8')
        renderer=screenshot(svg_path,png_path)
        rendered=png_path.read_bytes()
        metrics=compare_images(image,rendered)
        object_metrics=compare_objects(image,rendered,current,geometry)
        heatmap_path=job_dir/f'iter_{i+1}_color_heatmap.png'
        regions=region_comparison(image,rendered,current,geometry,heatmap_path=heatmap_path)
        rglobal=regions['global']
        object_focus=sorted((o['color_score'] for o in regions['objects']))[:6]
        object_color=sum(object_focus)/max(1,len(object_focus)) if object_focus else rglobal['color_score']
        # New comparison combines spatial color + structure. Unlike prior
        # metrics, the report also exposes raw DeltaE, not just a percentage.
        selection_score=(.25*metrics['score']+.40*rglobal['color_score']
                         +.20*rglobal['edge_f1']+.15*object_color)
        item={'iteration':i+1,'metrics':metrics,'region_metrics':regions,
              'object_metrics':object_metrics,'traced_objects':list(geometry['objects']),
              'geometry_warnings':geometry['warnings'], 'renderer':renderer,
              'scene_objects':len(current['objects']),
              'selection_score':round(selection_score,4), 'color_refinement':refine_info}
        history.append(item)
        if best is None or selection_score>best['selection_score']:
            best={'scene':current,'geo':geometry,'color':color_ir,'detail':details_ir,
                  'render':png_path,'svg':svg_path,'heatmap':heatmap_path,
                  'regions':regions,'score':metrics['score'],
                  'selection_score':selection_score,'iteration':i+1}
        if i+1>=iterations or demo or (rglobal['mean_delta_e00']<4 and rglobal['edge_f1']>.87):
            break
        progress('reviewing',pct+19,'分区色差与对象几何评估；执行局部返工')
        feedback=review(client,image,rendered,current,
                        {**metrics,'object_metrics':object_metrics,'region_metrics':regions})
        dump(job_dir/f'review_{i+1}.json',feedback)
        next_scene,changed=apply_patches(current,feedback.get('patches',[]))
        history[-1]['repaired_objects']=changed
        current=next_scene
        if color_enabled:
            # Deterministic additional local colour sampling. No full rewrite.
            color_ir,refine_info=refine_worst_band(image,color_ir,regions)
            history[-1]['color_refinement_next']=refine_info
        if not changed and not (refine_info or {}).get('modified'):
            break
    if best is None:
        raise RuntimeError('Render pipeline produced no output')
    (job_dir/'preview.svg').write_text(best['svg'].read_text(encoding='utf-8'),encoding='utf-8')
    (job_dir/'render.png').write_bytes(best['render'].read_bytes())
    (job_dir/'color_heatmap.png').write_bytes(best['heatmap'].read_bytes())
    dump(job_dir/'scene.json',best['scene'])
    dump(job_dir/'geometry.json',best['geo'])
    dump(job_dir/'color.json',best['color'])
    dump(job_dir/'detail.json',best['detail'])
    dump(job_dir/'region_metrics.json',best['regions'])
    report={'version':'0.5','mode':'demo' if demo else 'vlm',
            'best_score':round(best['score'],4),
            'best_selection_score':round(best['selection_score'],4),
            'best_iteration':best['iteration'],
            'geometry_enabled':geometry_enabled,'color_enabled':color_enabled,
            'detail_enabled':detail_enabled,
            'traced_objects':list(best['geo'].get('objects',{})),
            'geometry_warnings':best['geo'].get('warnings',[]),
            'color_statistics':best['color'].get('statistics',{}),
            'detail_statistics':best['detail'].get('statistics',{}),
            'best_region_metrics':best['regions'],
            'iterations_completed':len(history),'history':history,
            'scene_objects':len(best['scene']['objects']),
            'notice':'DeltaE00 is a perceptual color difference, not a similarity percentage. '
                     'Continuous gradient sampling directly traces source colours; it is not a '
                     'generative re-drawing or semantic part reconstruction. '
                     'Demo uses a fixed scene, and real model connectivity is unverified.'}
    dump(job_dir/'report.json',report)
    progress('completed',100,'完成：Scene、Geometry、Color、Detail IR、PNG、SVG、区域色差图已保存')
    return report

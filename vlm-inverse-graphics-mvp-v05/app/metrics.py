"""Cheap explainable low-frequency and edge comparison metrics; NOT a similarity guarantee."""
from __future__ import annotations
from io import BytesIO
import numpy as np
from PIL import Image, ImageFilter


def compare_images(original: bytes, rendered: bytes) -> dict:
    def rgb(data):
        return Image.open(BytesIO(data)).convert('RGB').resize((256,256),Image.Resampling.LANCZOS)
    src, dst = rgb(original),rgb(rendered)
    blur_a = np.asarray(src.filter(ImageFilter.GaussianBlur(12)),dtype=np.float32)
    blur_b = np.asarray(dst.filter(ImageFilter.GaussianBlur(12)),dtype=np.float32)
    global_score=float(np.clip(1-np.mean(np.abs(blur_a-blur_b))/255,0,1))
    pix_a=np.asarray(src,dtype=np.float32)
    pix_b=np.asarray(dst,dtype=np.float32)
    color_score=float(np.clip(1-np.mean(np.abs(pix_a-pix_b))/255,0,1))
    def edge(im):
        grey=im.convert('L').filter(ImageFilter.FIND_EDGES)
        return np.asarray(grey,dtype=np.uint8)>55
    ea,eb=edge(src),edge(dst)
    # 3x3 tolerance handles small registration errors.
    thick=Image.fromarray((eb*255).astype('uint8')).filter(ImageFilter.MaxFilter(5))
    padded_b=np.asarray(thick)>0
    precision=float((ea & padded_b).sum()/max(1,int(ea.sum())))
    padded_a=np.asarray(Image.fromarray((ea*255).astype('uint8')).filter(ImageFilter.MaxFilter(5)))>0
    recall=float((eb & padded_a).sum()/max(1,int(eb.sum())))
    edge_score=(1.0 if not ea.any() and not eb.any()
                else 2*precision*recall/max(1e-8,precision+recall))
    score=0.5*global_score+0.3*color_score+0.2*edge_score
    return {"score":round(score,4),"composition":round(global_score,4),
            "color":round(color_score,4),"edge":round(edge_score,4),
            "note":"Heuristic reference metric; not semantic fidelity or pixel-level equivalence"}


def compare_objects(original: bytes, rendered: bytes, scene: dict, geometry: dict | None = None) -> list[dict]:
    """Scores inside each object ROI, instead of letting large flat background dominate.

    Local scores are heuristic; overlapping subjects may be affected by occluders.
    """
    geo_ids=set((geometry or {}).get('objects', {}))
    a=Image.open(BytesIO(original)).convert('RGB').resize((1000,1000),Image.Resampling.LANCZOS)
    b=Image.open(BytesIO(rendered)).convert('RGB').resize((1000,1000),Image.Resampling.LANCZOS)
    results=[]
    for obj in scene['objects']:
        x,y,w,h=obj['bbox']
        box=(max(0,int(x)),max(0,int(y)),min(1000,int(x+w)),min(1000,int(y+h)))
        if box[2]-box[0]<10 or box[3]-box[1]<10:
            continue
        out_a,out_b=BytesIO(),BytesIO()
        a.crop(box).resize((256,256)).save(out_a,format='PNG')
        b.crop(box).resize((256,256)).save(out_b,format='PNG')
        m=compare_images(out_a.getvalue(),out_b.getvalue())
        results.append({'id':obj['id'],'type':obj['type'],
                        'render_mode':'traced' if obj['id'] in geo_ids else 'procedural',
                        **{k:m[k] for k in ('score','composition','color','edge')}})
    return sorted(results,key=lambda r:r['score'])

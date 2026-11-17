"""Spatially aligned CIEDE2000 / edge metrics by area and visible object.

A good global histogram is insufficient when colours are in the wrong places.
No invented ground-truth instance masks: results say whether an actual traced
mask or just an approximate ROI was used.
"""
from __future__ import annotations
import io

import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from .segmentation import visible_instance_masks


def _rgb(data: bytes, size: int) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert('RGB').resize((size,size),Image.Resampling.LANCZOS))


def region_comparison(original: bytes, rendered: bytes, scene: dict,
                      geometry: dict | None = None, heatmap_path=None,
                      size: int = 256) -> dict:
    src=_rgb(original,size)
    dst=_rgb(rendered,size)
    lab_src=rgb2lab(src)
    lab_dst=rgb2lab(dst)
    delta=deltaE_ciede2000(lab_src, lab_dst).astype(np.float32)
    src_edges=cv2.Canny(cv2.cvtColor(src,cv2.COLOR_RGB2GRAY),60,125)>0
    dst_edges=cv2.Canny(cv2.cvtColor(dst,cv2.COLOR_RGB2GRAY),60,125)>0
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    thick_a=cv2.dilate(src_edges.astype('uint8'),k)>0
    thick_b=cv2.dilate(dst_edges.astype('uint8'),k)>0

    def score(mask):
        count=int(np.count_nonzero(mask))
        if count<12:
            return None
        error=float(delta[mask].mean())
        a=src_edges&mask;b=dst_edges&mask
        # Precision and recall are symmetric here; both edges are measured in
        # a consistent mask, not inflated by unrelated objects outside it.
        matched_a=int((a&thick_b).sum());matched_b=int((b&thick_a).sum())
        p=matched_a/max(1,int(a.sum()))
        r=matched_b/max(1,int(b.sum()))
        edge=1.0 if not a.any() and not b.any() else 2*p*r/max(1e-8,p+r)
        return {'mean_delta_e00':round(error,3),
                'color_score':round(float(np.clip(1-error/50,0,1)),4),
                'edge_f1':round(float(edge),4),
                'pixels':count}

    bands=[('sky',0,.31),('midground',.31,.52),('foreground_water',.52,1.0)]
    regions=[]
    for name,y0,y1 in bands:
        mask=np.zeros((size,size),dtype=bool)
        mask[round(y0*size):round(y1*size),:]=True
        regions.append({'id':name,'mask_type':'spatial_band',**score(mask)})
    masks=visible_instance_masks(scene,geometry or {},size)
    obj_results=[]
    for obj in scene.get('objects',[]):
        objid=obj['id']
        if objid in masks and np.count_nonzero(masks[objid])>=12:
            mask=masks[objid]>0
            mask_type='traced_visible_estimate'
        else:
            x,y,w,h=obj['bbox'];mask=np.zeros((size,size),dtype=bool)
            x0=max(0,min(size,round(x*size/1000)))
            y0=max(0,min(size,round(y*size/1000)))
            x1=max(x0,min(size,round((x+w)*size/1000)))
            y1=max(y0,min(size,round((y+h)*size/1000)))
            mask[y0:y1,x0:x1]=True
            mask_type='bbox_fallback'
        measured=score(mask)
        if measured:
            obj_results.append({'id':objid,'type':obj['type'],'mask_type':mask_type,**measured})
    obj_results.sort(key=lambda o:o['mean_delta_e00'],reverse=True)
    if heatmap_path:
        cm=cv2.applyColorMap(np.uint8(np.clip(delta/60,0,1)*255),cv2.COLORMAP_TURBO)
        cm=cv2.resize(cm,(1000,1000),interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(heatmap_path),cm)
    return {'version':'0.4', 'metric':'CIEDE2000 + edge F1 (spatially aligned)',
            'global':score(np.ones((size,size),dtype=bool)),
            'regions':regions,'objects':obj_results,
            'note':'DeltaE00 is a colour difference, not a fidelity percentage. '
                   'BBox fallbacks can include surrounding pixels; visibility masks are GrabCut estimates.'}

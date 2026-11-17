"""Source-grounded Geometry IR: image -> bounded mask -> editable numeric vector paths.

No source image is embedded in SVG. GrabCut is an offline MVP fallback for
instance masks; foreground/background point hints improve overlap handling.
For high-quality production use, substitute a SAM mask adapter at this boundary.
"""
from __future__ import annotations

import io
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image

TRACE_KINDS = frozenset({'koi', 'lotus', 'pine', 'house', 'bridge', 'lily_pad', 'rock'})


def _points_from_contour(contour: np.ndarray, sx: float, sy: float, ox: int, oy: int, epsilon: float = 1.0) -> list[list[float]]:
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) > 650:
        approx = approx[::int(np.ceil(len(approx) / 650))]
    return [[round((float(px) + ox) * sx, 2), round((float(py) + oy) * sy, 2)] for px, py in approx]


def _point_in_roi(p, ox: int, oy: int, source_w: int, source_h: int) -> tuple[int, int]:
    return round(p[0] * source_w / 1000) - ox, round(p[1] * source_h / 1000) - oy


def _contours(mask: np.ndarray, minimum_area: float, max_regions: int = 220):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted((c for c in contours if cv2.contourArea(c) >= minimum_area),
                  key=cv2.contourArea, reverse=True)[:max_regions]


def _grabcut(image: np.ndarray, item: dict, all_objects: list[dict]) -> tuple[np.ndarray, int, int, dict]:
    height, width = image.shape[:2]
    x, y, w, h = item['bbox']
    bx, by, bw, bh = [round(v * s) for v, s in zip((x, y, w, h), (width/1000, height/1000, width/1000, height/1000))]
    margin = max(12, round(max(bw, bh) * 0.10))
    x0, y0 = max(0, bx-margin), max(0, by-margin)
    x1, y1 = min(width, bx+bw+margin), min(height, by+bh+margin)
    if x1-x0 < 16 or y1-y0 < 16:
        raise ValueError('Object too small for instance extraction')
    crop = image[y0:y1, x0:x1].copy()
    mh, mw = crop.shape[:2]
    gc_mask = np.full((mh, mw), cv2.GC_BGD, np.uint8)
    left, right = max(0, bx-x0), min(mw, bx+bw-x0)
    top, bottom = max(0, by-y0), min(mh, by+bh-y0)
    gc_mask[top:bottom, left:right] = cv2.GC_PR_BGD
    cv2.ellipse(gc_mask, (round(bx+bw*.5-x0), round(by+bh*.5-y0)),
                (max(2,round(bw*.32)), max(2,round(bh*.28))), 0, 0, 360,
                cv2.GC_PR_FGD, -1)

    hint = item.get('geometry_hint', {})
    fg_points = hint.get('foreground_points', [])
    bg_points = list(hint.get('background_points', []))
    if not fg_points:
        fg_points = [[x+w*.48, y+h*.5]]
    # Other same-type objects provide negative hints when their ROIs overlap.
    # Do not assume image bounding boxes define a valid instance boundary.
    for other in all_objects:
        if other['id'] == item['id'] or other['type'] != item['type']:
            continue
        other_hint = other.get('geometry_hint', {})
        others = other_hint.get('foreground_points') or [[other['bbox'][0]+other['bbox'][2]*.5,
                                                         other['bbox'][1]+other['bbox'][3]*.5]]
        for p in others:
            px,py = _point_in_roi(p, x0,y0,width,height)
            if 0 <= px < mw and 0 <= py < mh:
                bg_points.append(p)

    # The hint lists contain coordinates, not executable SVG/HTML.
    for p in fg_points:
        px, py = _point_in_roi(p,x0,y0,width,height)
        if 0 <= px < mw and 0 <= py < mh:
            cv2.circle(gc_mask,(px,py),max(3,round(min(bw,bh)*.025)),cv2.GC_FGD,-1)
    for p in bg_points:
        px,py = _point_in_roi(p,x0,y0,width,height)
        if 0 <= px < mw and 0 <= py < mh:
            cv2.circle(gc_mask,(px,py),max(4,round(min(bw,bh)*.037)),cv2.GC_BGD,-1)

    bgd, fgd = np.zeros((1,65),np.float64), np.zeros((1,65),np.float64)
    cv2.grabCut(crop, gc_mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    mask = np.uint8((gc_mask==cv2.GC_FGD)|(gc_mask==cv2.GC_PR_FGD))*255
    # Remove islands and fill tiny holes but preserve the fins and tail contour.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for idx in range(1,count):
        if stats[idx, cv2.CC_STAT_AREA] >= max(20, bw*bh*.012):
            keep[labels == idx] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    keep = cv2.morphologyEx(keep,cv2.MORPH_CLOSE,kernel,iterations=1)
    # For a pale gold fish crossing a dark fish, GrabCut often merges both.
    # Restrict to the locally light material and recover narrow ink outlines
    # with morphology. This is object-specific *geometry*, not a fish template.
    if item['type']=='koi' and item.get('params',{}).get('pattern')=='gold':
        light = cv2.GaussianBlur(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(9,9),0)
        keep = cv2.bitwise_and(keep,np.uint8(light>124)*255)
        keep = cv2.morphologyEx(keep,cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),iterations=1)
        cs=_contours(keep, minimum_area=max(45,bw*bh*.02),max_regions=5)
        keep[:]=0
        if cs:
            cv2.drawContours(keep,[cs[0]],-1,255,-1)
    if item['type']=='koi' and item.get('params',{}).get('pattern')=='red_white':
        # The original pond is blue/green-gray but the fish is warm red/cream.
        # Prevent a broad GrabCut water region from becoming a giant cream fin.
        rr,gg,bb = [crop[:,:,i].astype(np.int16) for i in (2,1,0)]
        warm=np.uint8((rr >= gg-5) & (rr >= bb+7))*255
        keep=cv2.bitwise_and(keep,warm)
        keep=cv2.morphologyEx(keep,cv2.MORPH_CLOSE,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        cs=_contours(keep,minimum_area=max(70,bw*bh*.035),max_regions=6)
        if cs:
            keep[:]=0
            cv2.drawContours(keep,[cs[0]],-1,255,-1)
    ratio = int(np.count_nonzero(keep)) / max(1, bw*bh)
    span_y, span_x = np.where(keep > 0)
    extent_x = (int(span_x.max())-int(span_x.min())+1) / max(1,bw) if len(span_x) else 0
    extent_y = (int(span_y.max())-int(span_y.min())+1) / max(1,bh) if len(span_y) else 0
    min_extent = .64 if item['type'] in ('bridge','house') else .42
    if ratio < .045 or ratio > .91 or extent_x < min_extent or extent_y < .26:
        raise ValueError(f'GrabCut failed shape coverage: area={ratio:.3f}, x={extent_x:.3f}, y={extent_y:.3f}')
    return keep, x0, y0, {'area_ratio':round(ratio,3),'seed_points':len(fg_points),'background_points':len(bg_points)}


def _colored_regions(rgb: np.ndarray, mask: np.ndarray, sx: float, sy: float,
                     ox: int, oy: int, max_regions: int = 1400) -> tuple[str, list[dict]]:
    # Quantization is computed from the subject pixels, NOT from surrounding water.
    yy, xx = np.where(mask > 0)
    if len(yy) < 30:
        raise ValueError('Not enough pixels for palette quantization')
    pixels = rgb[yy,xx]
    samples = pixels[::max(1,len(pixels)//8000)]
    # Median-cut is deterministic; its output colors remain ordinary editable SVG fills.
    swatches = Image.fromarray(samples.reshape(-1,1,3),'RGB').quantize(
        colors=12, method=Image.Quantize.MEDIANCUT)
    palette = np.asarray(swatches.getpalette(), dtype=np.uint8).reshape(-1,3)
    palette = np.unique(palette,axis=0)
    if len(palette)>12:  # Pillow's unused palette entries may be repeated or black.
        palette = palette[:12]
    active_rgb = pixels.astype(np.int16)
    # Explicit color nearest-neighbor assignment for foreground pixels only.
    difference = active_rgb[:,None,:] - palette[None,:,:].astype(np.int16)
    color_idx = np.argmin(np.sum(difference.astype(np.int32)**2,axis=2),axis=1)
    counts = np.bincount(color_idx,minlength=len(palette))
    dominant = int(np.argmax(counts))
    base = '#'+''.join(f'{c:02X}' for c in palette[dominant])
    index_map = np.full(mask.shape,255,np.uint8)
    index_map[yy,xx] = color_idx.astype(np.uint8)
    regions=[]
    for idx in np.argsort(-counts):
        idx=int(idx)
        if idx==dominant or counts[idx]<24:
            continue
        mono=np.uint8((index_map==idx))*255
        # Reconnect color fragments separated by aliasing while retaining ink edges.
        cs = _contours(mono, minimum_area=max(6, mask.size*.000028), max_regions=max_regions)
        fill = '#'+''.join(f'{c:02X}' for c in palette[idx])
        for contour in cs:
            if len(regions)>=max_regions:
                break
            pts=_points_from_contour(contour,sx,sy,ox,oy,epsilon=.55)
            if len(pts)>=3:
                regions.append({'fill':fill,'points':pts})
        if len(regions)>=max_regions:
            break
    # Larger regions are emitted first, then fine hatching and scale marks.
    regions.sort(key=lambda r: -len(r['points']))
    return base,regions


def _material_regions(rgb: np.ndarray, mask: np.ndarray, item: dict,
                      sx: float, sy: float, ox: int, oy: int) -> tuple[str,list[dict]]:
    """Classify color materials, then vectorize fine ink/scales as separate paths."""
    pat=item.get('params',{}).get('pattern','red_white')
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    sat=hsv[:,:,1]
    hue=hsv[:,:,0]
    regions=[]
    if pat=='gold':
        base='#E6CA96'
    elif pat=='black':
        base='#36332B'
    elif pat=='red_white':
        base='#F1DCC0'
    else:
        base='#EADDC6'

    def emit(binary, fill, minimum_area, limit=700, epsilon=.50):
        binary=np.uint8((binary>0)&(mask>0))*255
        contours=_contours(binary,minimum_area,max_regions=limit)
        for c in contours:
            pts=_points_from_contour(c,sx,sy,ox,oy,epsilon=epsilon)
            if len(pts)>=3:
                regions.append({'fill':fill,'points':pts})

    if pat=='red_white':
        # Larger warm color fields are recovered first; texture goes on top.
        red=np.uint8(((hue<=21)|(hue>=175))&(sat>66)&(rgb[:,:,0]>rgb[:,:,1]*1.10))*255
        red=cv2.morphologyEx(red,cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
        emit(red,'#D36644',25,limit=130)
        # Red/cream boundary shade, preserving the woodblock-style two-tone body.
        orange=np.uint8((hue<22)&(sat>112)&(rgb[:,:,0]>155))*255
        orange=cv2.morphologyEx(orange,cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        emit(orange,'#C94F30',17,limit=240)
    elif pat=='gold':
        gold_shadow=np.uint8((gray<170)&(gray>95)&(sat>35))*255
        emit(gold_shadow,'#C49E6B',12,limit=240)
    elif pat=='black':
        lit=np.uint8((gray>110)&(gray<215))*255
        emit(lit,'#9F8F70',6,limit=1200,epsilon=.32)

    if pat!='black':
        # Sub-pixel dark woodblock strokes and fish scales.
        ink=np.uint8((gray < 98) if pat=='gold' else ((gray < 88)&(rgb[:,:,0]<110)))*255
        emit(ink,'#43382B',2.2,limit=1200,epsilon=.36)
    return base,regions


def _ink_strokes(rgb: np.ndarray, mask: np.ndarray, item: dict,
                 sx: float, sy: float, ox: int, oy: int) -> list[dict]:
    """Vectorize source-specific scale and contour edges as independent paths."""
    if item['type']!='koi':
        return []
    grey=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    canny=cv2.Canny(grey,60,130)
    interior=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    canny=cv2.bitwise_and(canny,interior)
    cs,_=cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cs=sorted((c for c in cs if cv2.arcLength(c,True)>12),
              key=lambda c:cv2.arcLength(c,True),reverse=True)[:450]
    tone=('#B7A486' if item['params']['pattern']=='black' else '#4A3A2D')
    strokes=[]
    for c in cs:
        approx=cv2.approxPolyDP(c,.72,False)
        pts=approx.reshape(-1,2)
        if 3<=len(pts)<=600:
            strokes.append({'color':tone,'width':0.72,'points':[
                [round((float(px)+ox)*sx,2),round((float(py)+oy)*sy,2)] for px,py in pts]})
    return strokes


def _extract_structural_parts(rgb: np.ndarray, mask: np.ndarray, item: dict,
                              sx: float, sy: float, ox: int, oy: int) -> list[dict]:
    """Source-grounded line segments, available as semantic-ish editable parts.

    The detector does not claim to identify every semantic element. For example
    rails / grids are candidate structure lines, validated against source edge.
    """
    kind=item['type']
    if kind not in ('house','bridge','pine','lotus','lily_pad','rock'):
        return []
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),.7),55,125)
    # Avoid free-floating lines outside the extracted foreground.
    inside=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    edges=cv2.bitwise_and(edges,inside)
    min_len=max(9,int(min(rgb.shape[:2])*.10))
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=18,minLineLength=min_len,maxLineGap=7)
    if lines is None:
        return []
    kind_map={'house':'house_grid_or_beam','bridge':'rail_or_pillar','pine':'branch_candidate',
              'lotus':'petal_crease','lily_pad':'leaf_vein','rock':'rock_crease'}
    out=[]
    for i,line in enumerate(lines[:,0][:115]):
        x1,y1,x2,y2=map(int,line)
        out.append({'id':f'{kind_map[kind]}_{i}',
                    'points':[[round((x1+ox)*sx,2),round((y1+oy)*sy,2)],
                              [round((x2+ox)*sx,2),round((y2+oy)*sy,2)]],
                    'stroke':'#384D48'})
    return out


def trace_object(image: np.ndarray, item: dict, all_objects: list[dict]) -> dict:
    mask, ox, oy, info = _grabcut(image,item,all_objects)
    height,width = image.shape[:2]
    sx,sy=1000/width,1000/height
    contours = _contours(mask, max(35.0, mask.size*.00017), max_regions=12)
    if not contours:
        raise ValueError('No visible contour found')
    outline=[_points_from_contour(c,sx,sy,ox,oy,epsilon=.95) for c in contours]
    outline=[p for p in outline if len(p)>=3]
    if not outline:
        raise ValueError('No valid outline path')
    crop=image[oy:oy+mask.shape[0],ox:ox+mask.shape[1]]
    rgb=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    if item['type']=='koi':
        base,regions=_material_regions(rgb,mask,item,sx,sy,ox,oy)
    else:
        base,regions=_colored_regions(rgb,mask,sx,sy,ox,oy)
    strokes=_ink_strokes(rgb,mask,item,sx,sy,ox,oy)
    parts=_extract_structural_parts(rgb,mask,item,sx,sy,ox,oy)
    return {'mode':'traced','method':'opencv_grabcut_material_trace', 'bbox':item['bbox'],
            'outline':outline,'base_fill':base,'regions':regions,'strokes':strokes,'parts':parts,
            'statistics':{**info,'region_count':len(regions),'stroke_count':len(strokes),'part_count':len(parts)}}


def extract_geometry(image_bytes: bytes, scene: dict, enabled: bool = True) -> dict:
    """Return separate JSON-serializable Geometry IR, with safe per-object fallback."""
    if not enabled:
        return {'version':'0.3','mode':'procedural','objects':{},'warnings':[]}
    image_array=np.asarray(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    img=cv2.cvtColor(image_array,cv2.COLOR_RGB2BGR)
    configured=os.getenv('GEOMETRY_TRACE_KINDS', 'koi,lotus,pine,house,bridge,lily_pad,rock')
    allowed={s.strip() for s in configured.split(',') if s.strip()}
    if not allowed:
        allowed=set(TRACE_KINDS)
    objects={}
    warnings=[]
    for item in scene['objects']:
        hint=item.get('geometry_hint',{})
        mode=hint.get('mode','auto')
        if mode=='template' or (mode=='auto' and item['type'] not in allowed):
            continue
        try:
            objects[item['id']]=trace_object(img,item,scene['objects'])
        except (ValueError,cv2.error) as exc:
            # A bad mask never destroys the scene. The semantic object template stays editable.
            warnings.append({'id':item['id'],'reason':str(exc)[:180],'fallback':'procedural'})
    return {'version':'0.3','mode':'hybrid','objects':objects,'warnings':warnings}

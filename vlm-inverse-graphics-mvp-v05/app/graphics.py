"""Deterministic SVG graphics engine. No model-generated JS/SVG is executed.

All coordinates are normalized to 1000x1000. Each semantic object is a small
procedural drawing program whose parameters are supplied by the Scene IR.
"""
from __future__ import annotations
from html import escape
import os
import math
import random
from .schema import normalize_scene


def ellipse(cx, cy, rx, ry, fill, stroke="none", sw=2, extra=""):
    return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'


def path(d, fill="none", stroke="none", sw=2, extra=""):
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" {extra}/>'


def rand_of(obj):
    return random.Random(obj["params"]["seed"])


def object_svg(o):
    kind, fill, ink = o["type"], o["fill"], o["stroke"]
    rng = rand_of(o)
    result = []
    if kind == "sun":
        result.append('<circle cx="500" cy="500" r="475" fill="'+fill+'"/>')
        for i in range(9):
            y = 220 + 57*i
            result.append(path(f"M {160+i%3*30} {y} Q 500 {y+5} {875-i%4*30} {y-2}",
                               stroke="#BC614D", sw=4, extra='opacity="0.21"'))
    elif kind == "mountain":
        result.append(path("M0 640 Q110 540 195 466 Q320 335 430 380 Q535 460 635 408 Q765 370 860 520 Q950 585 1000 565 L1000 1000 L0 1000Z", fill, ink, 12))
        result.append(path("M155 505 Q275 415 407 415 M450 470 Q545 515 655 447 M720 458 Q850 495 905 550",stroke="#3D5957",sw=10,extra='opacity="0.55"'))
        for i in range(22):
            x = rng.randint(100,900); y = rng.randint(540,890)
            result.append(path(f"M{x} {y} q{rng.randint(10,40)} {-rng.randint(12,36)} {rng.randint(20,64)} {-rng.randint(10,18)}", stroke="#456A5B", sw=2, extra='opacity="0.55"'))
    elif kind == "pond":
        result.append(path("M0 0 Q110 78 270 25 Q500 12 725 34 Q855 10 1000 45 L1000 1000 L0 1000Z", fill, "#435E60", 8))
        result.append(path("M0 38 Q180 110 335 52 Q710 90 1000 40",stroke="#587471",sw=8,extra='opacity="0.7"'))
        for i in range(24):
            x=rng.randint(0,1000); y=rng.randint(140,940); w=rng.randint(30,125)
            result.append(path(f"M {x} {y} q {w//2} {-rng.randint(6,22)} {w} 0",stroke="#526C68",sw=rng.randint(2,5),extra='opacity="0.37"'))
    elif kind == "bridge":
        a=o["params"]["arch"]
        rise=int(a*750)
        base=625
        upper=base-rise
        result.append(path(f"M0 {base+80} Q500 {upper+68} 1000 {base+80} L1000 {base+190} Q500 {upper+178} 0 {base+190}Z", "#6A7972", ink, 10))
        result.append(path(f"M-10 {base+46} Q500 {upper+36} 1010 {base+46}",stroke=fill,sw=32))
        result.append(path(f"M-10 {base-92} Q500 {upper-112} 1010 {base-92}",stroke=fill,sw=23))
        result.append(path(f"M-10 {base-12} Q500 {upper-36} 1010 {base-12}",stroke=fill,sw=12))
        for x in range(25,1000,125):
            shift=(1-((x-500)/500)**2)*rise
            top=base-92-shift
            result.append(path(f"M{x} {top-35} L{x} {top+148}",stroke=ink,sw=25))
            result.append(path(f"M{x} {top-35} L{x} {top+140}",stroke=fill,sw=17))
            result.append(ellipse(x,top-38,14,19,ink))
        for x in (265,700):
            result.append(path(f"M{x} {base+160} L{x} 1000",stroke=ink,sw=48))
            result.append(path(f"M{x} {base+160} L{x} 1000",stroke="#8D9690",sw=36))
    elif kind == "house":
        result.append(path("M130 395 L790 395 L785 900 L120 900 Z", "#C1B795",ink,12))
        result.append(path("M35 390 L215 130 L710 125 L970 390 Q770 425 35 390Z", "#46524A",ink,15))
        for i in range(13):
            x=220+i*40
            result.append(path(f"M{x} 155 L{x-40} 375",stroke="#778074",sw=7,extra='opacity="0.7"'))
        for x in range(185,775,150):
            result.append('<rect x="'+str(x)+'" y="420" width="128" height="370" fill="#D8CCAA" stroke="'+ink+'" stroke-width="9"/>')
            for k in range(1,4):
                result.append(path(f"M{x+28*k} 430 V785",stroke="#717A72",sw=5))
            for y in (520,610,700):
                result.append(path(f"M{x+4} {y} H{x+125}",stroke="#717A72",sw=5))
        result.append(path("M120 805 H820 M140 870 H840",stroke=ink,sw=18))
        for x in range(180,830,115):
            result.append(path(f"M{x} 805 V910",stroke=ink,sw=10))
    elif kind == "pine":
        result.append(path("M470 1000 Q385 700 455 525 Q515 350 630 115", stroke=ink,sw=76))
        result.append(path("M470 1000 Q385 700 455 525 Q515 350 630 115",stroke="#9F6041",sw=55))
        result.append(path("M451 590 Q250 400 170 280 M480 465 Q695 340 780 270 M548 320 Q415 240 325 205",stroke=ink,sw=40))
        result.append(path("M451 590 Q250 400 170 280 M480 465 Q695 340 780 270 M548 320 Q415 240 325 205",stroke="#986548",sw=24))
        for cx,cy,rx,ry in [(625,130,260,90),(330,205,200,80),(180,280,165,90),(750,275,195,85),(470,350,185,73)]:
            result.append(path(f"M{cx-rx} {cy+35} Q{cx-rx} {cy-ry} {cx} {cy-ry} Q{cx+rx} {cy-ry} {cx+rx} {cy+25} Q{cx} {cy+ry//2} {cx-rx} {cy+35}Z", "#385B53",ink,11))
            for _ in range(25):
                px=rng.randint(cx-rx+25,cx+rx-25); py=rng.randint(cy-ry+15,cy+23)
                result.append(path(f"M{px} {py+13} l{rng.randint(-18,18)} {-rng.randint(7,24)}",stroke="#223E3F",sw=3,extra='opacity="0.7"'))
    elif kind == "koi":
        pat=o["params"]["pattern"]
        body={"gold":"#E9D3A0","red_white":"#F1E5C8","black":"#2C3030","white":"#E5DDD2"}[pat]
        patchcol={"gold":"#C38A53","red_white":"#C95432","black":"#98896B","white":"#AC9A80"}[pat]
        flip='translate(1000 0) scale(-1 1)' if o["params"]["facing"]=="right" else ''
        result.append(f'<g transform="{flip}">')
        bodypath="M105 500 C135 290 370 265 635 365 Q730 415 806 460 Q865 410 960 240 Q940 435 855 510 Q930 615 975 782 Q845 680 785 570 C580 765 270 735 130 590 Q95 555 105 500Z"
        uid=escape(o["id"])
        result.append(f'<defs><clipPath id="clip-{uid}">{path(bodypath,fill="white")}</clipPath></defs>')
        result.append(path("M360 325 Q440 120 560 170 L650 380 M420 660 Q420 815 575 850 L610 668",fill="#C6AB82",stroke=ink,sw=10))
        result.append(path(bodypath,body,ink,15))
        result.append(f'<g clip-path="url(#clip-{uid})">')
        if pat == "red_white":
            for d in ("M310 240 Q380 410 310 690 L550 750 Q460 500 570 260Z","M670 280 Q650 500 730 640 L870 665 L830 290Z"):
                result.append(path(d,patchcol,sw=0))
        elif pat == "gold":
            result.append(path("M300 250 Q440 500 380 720 L600 750 Q520 480 690 350Z",patchcol,sw=0,extra='opacity="0.2"'))
        elif pat == "black":
            result.append(path("M110 455 Q410 300 700 420 L805 530 Q450 715 130 570Z", "#353631",sw=0))
        for row in range(6):
            for col in range(16):
                x=210+col*42+(row%2)*20; y=350+row*46
                result.append(path(f"M{x-21} {y-14} Q{x+4} {y+15} {x+22} {y-14}",stroke="#796B50" if pat!='black' else "#A9976D",sw=4,extra='opacity="0.75"'))
        result.append('</g>')
        result.append(ellipse(172,493,19,23,"#EBDEC4",ink,8))
        result.append(ellipse(179,493,8,12,"#22252A"))
        result.append(path("M105 528 Q55 555 35 536 M125 550 Q70 610 45 605",stroke=ink,sw=7))
        result.append(path("M110 510 Q80 503 66 513",stroke=ink,sw=7))
        result.append('</g>')
    elif kind == "lotus":
        result.append(ellipse(500,830,445,130,"#567F68",ink,15))
        result.append(path("M500 830 L890 845 M500 830 L235 765 M500 830 L490 945",stroke="#244D47",sw=7))
        for i in range(11):
            ang=-1.7+i*0.33
            bx=500+math.sin(ang)*220
            by=655+math.cos(ang)*60
            result.append(path(f"M500 730 Q{bx-175} {by-120} {bx:.1f} {by-300} Q{bx+160} {by-120} 500 730Z", "#F0CCB0" if i%2 else "#F5DFC2", "#B86753", 10))
        result.append(ellipse(500,720,85,50,"#C89750",ink,9))
        for x in range(450,552,24):
            result.append('<circle cx="'+str(x)+'" cy="705" r="8" fill="#63805B"/>')
    elif kind == "lily_pad":
        result.append(path("M550 500 L920 330 Q880 130 595 128 Q200 100 75 390 Q30 650 310 785 Q660 850 910 590 Q730 590 550 500Z",fill,ink,19))
        result.append(path("M550 500 L310 220 M550 500 L170 460 M550 500 L345 715 M550 500 L750 765",stroke="#436653",sw=8))
    elif kind == "reed":
        for i in range(o["params"]["density"]+12):
            x=rng.randint(70,930); y=1000
            tx=x+rng.randint(-420,390); ty=rng.randint(10,470)
            result.append(path(f"M{x} {y} Q{x+rng.randint(-90,90)} 590 {tx} {ty}",stroke="#345C4C" if i%3 else "#7C9167",sw=rng.randint(5,15)))
    elif kind == "rock":
        result.append(path("M80 820 Q10 680 205 520 Q180 400 360 320 Q560 220 680 400 Q890 370 950 610 Q1000 790 860 850 Q450 920 80 820Z", fill,ink,16))
        result.append(path("M205 520 Q390 600 430 800 M680 400 Q600 530 730 815",stroke="#4B605B",sw=9))
    elif kind == "ripple":
        for i in range(4):
            result.append(path(f"M{70+i*55} {500+i*30} C{230+i*35} {340+i*25} {770-i*35} {345+i*25} {930-i*55} {500+i*30}",stroke=fill,sw=17-i*2,extra='opacity="0.75"'))
    elif kind == "bush":
        for i in range(13):
            cx=rng.randint(40,960); cy=rng.randint(250,780)
            result.append(ellipse(cx,cy,rng.randint(100,210),rng.randint(120,230),fill,ink,8))
    elif kind == "cloud":
        for i in range(3):
            result.append(path(f"M{45+i*80} {400+i*110} Q250 {150+i*65} 500 {300+i*55} T{950-i*60} {350+i*90}",stroke=fill,sw=60,extra='opacity="0.55"'))
    return ''.join(result)


def points_to_svg_path(points, curved: bool = False) -> str:
    """Turn trusted CV-generated numeric control points into a closed editable SVG path.

    Catmull-Rom -> cubic Bezier on outer contours, fine polygon paths inside.
    This is not a pixel-embedded image or a model-provided raw SVG string.
    """
    import math
    clean=[]
    for pair in points[:700]:
        if (isinstance(pair, (list,tuple)) and len(pair)==2 and
            all(isinstance(v,(int,float)) and math.isfinite(v) and -50<=v<=1050 for v in pair)):
            clean.append((float(pair[0]),float(pair[1])))
    if len(clean)<3:
        return ''
    d=f'M{clean[0][0]:.2f},{clean[0][1]:.2f}'
    if not curved or len(clean)<5:
        return d+' '.join(f' L{x:.2f},{y:.2f}' for x,y in clean[1:])+' Z'
    # Tension is deliberately restrained to protect narrow fins and complex silhouettes.
    n=len(clean)
    for i in range(n):
        prev,p0,p1,nxt=clean[(i-1)%n],clean[i],clean[(i+1)%n],clean[(i+2)%n]
        tension=.14
        a=(p0[0]+(p1[0]-prev[0])*tension,p0[1]+(p1[1]-prev[1])*tension)
        b=(p1[0]-(nxt[0]-p0[0])*tension,p1[1]-(nxt[1]-p0[1])*tension)
        d+=f' C{a[0]:.2f},{a[1]:.2f} {b[0]:.2f},{b[1]:.2f} {p1[0]:.2f},{p1[1]:.2f}'
    return d+' Z'


def traced_object_svg(obj: dict, geo: dict) -> str:
    """One semantic group remains editable; no bitmap is used by the renderer."""
    gid=escape('trace-'+obj['id'],quote=True)
    outlines=[points_to_svg_path(points,curved=True) for points in geo['outline']]
    outlines=[p for p in outlines if p]
    if not outlines:
        return ''
    clips=''.join(path(d,fill='white',sw=0) for d in outlines)
    body=''.join(path(d,geo['base_fill'],sw=0) for d in outlines)
    parts=[]
    for region in geo['regions']:
        d=points_to_svg_path(region['points'],curved=False)
        if d:
            parts.append(path(d,region['fill'],sw=0))
    ink=[]
    for stroke in geo.get('strokes',[]):
        pts=stroke.get('points',[])
        d=points_to_svg_path(pts,curved=False)
        if d:
            # Closing a Canny contour path is harmless when the contour is a loop;
            # all its points and attributes are bounded by our CV geometry producer.
            ink.append(path(d,stroke=stroke.get('color','#483E35'),sw=stroke.get('width',.7),
                            extra='opacity="0.69"'))
    contours=''.join(path(d,stroke=obj['stroke'],sw=1.35,extra='opacity="0.9"') for d in outlines)
    return (f'<defs><clipPath id="{gid}">{clips}</clipPath></defs>'
            +body+f'<g clip-path="url(#{gid})">'+''.join(parts)+''.join(ink)+'</g>'+contours)


def scene_to_svg_legacy(scene: dict, geometry: dict | None = None) -> str:
    scene=normalize_scene(scene)
    geo_objects=geometry.get('objects',{}) if isinstance(geometry,dict) else {}
    blocks=['<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1000 1000" width="1000" height="1000">',
        '<defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#E9DEBB"/><stop offset="0.55" stop-color="#F2D6B7"/><stop offset="1" stop-color="#CFD7BE"/></linearGradient></defs>',
        f'<rect width="1000" height="1000" fill="{scene["background"]}"/>',
        '<rect width="1000" height="1000" fill="url(#sky)" opacity="0.65"/>']
    for obj in scene['objects']:
        x,y,w,h=obj['bbox']
        if obj['id'] in geo_objects:
            traced=traced_object_svg(obj,geo_objects[obj['id']])
            if traced:
                blocks.append(f'<g data-object-id="{escape(obj["id"],quote=True)}" data-render-mode="traced">{traced}</g>')
                continue
        cx=x+w/2; cy=y+h/2
        transform=(f'translate({cx} {cy}) rotate({obj["params"]["rotation"]}) '
                   f'translate({-w/2} {-h/2}) scale({w/1000:.6f} {h/1000:.6f})')
        blocks.append(f'<g data-object-id="{escape(obj["id"],quote=True)}" data-render-mode="procedural" transform="{transform}">{object_svg(obj)}</g>')
    blocks.append('<rect x="3" y="3" width="994" height="994" fill="none" stroke="#263238" stroke-width="6"/>')
    blocks.append('</svg>')
    return '\n'.join(blocks)


def _vector_path(points: list, closed: bool = True) -> str:
    """Only encode numeric CV-generated coordinates; never accept raw SVG from VLM."""
    import math
    clean=[]
    for p in points[:400]:
        if (isinstance(p,list) and len(p)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) and -20<=v<=1020 for v in p)):
            clean.append((float(p[0]),float(p[1])))
    if len(clean)<(3 if closed else 2):
        return ''
    return 'M'+' L'.join(f'{x:.2f},{y:.2f}' for x,y in clean)+(' Z' if closed else '')


def scene_to_svg(scene: dict, geometry: dict | None = None,
                 colors: dict | None = None, details: dict | None = None) -> str:
    """Source-guided SVG; sharp continuous colour plus editable geometry/ink.

    Unlike v0.3, never strokes superpixel region boundaries. Every band is a
    1px high horizontal gradients with edge-adaptive colour stops and no border.
    """
    if not colors or colors.get('mode') != 'continuous_scanline_gradients':
        return scene_to_svg_legacy(scene, geometry)
    scene=normalize_scene(scene)
    objects=(geometry or {}).get('objects',{})
    out=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" width="1000" height="1000">',
         f'<rect width="1000" height="1000" fill="{colors.get("background",scene["background"])}"/>',
         '<defs>']
    # SVG linear gradients are genuinely continuous in X. Narrow, edgeless
    # horizontal scanlines approximate continuity in Y at output resolution.
    # The rendering library anti-aliases the gradient stops, rather than
    # drawing thousands of flat-colour cells with visible outlines.
    rows = colors.get('bands', [])[:1000]
    for i, band in enumerate(rows):
        out.append(f'<linearGradient id="color-row-{i}" x1="0%" y1="0%" x2="100%" y2="0%">')
        for x, color in band.get('stops', [])[:400]:
            if not isinstance(color,str) or len(color)!=7 or not color.startswith('#') or any(ch not in '0123456789abcdefABCDEF' for ch in color[1:]):
                continue
            pct=max(0.0,min(100.0,float(x)))
            out.append(f'<stop offset="{pct:.3f}%" stop-color="{color}"/>')
        out.append('</linearGradient>')
    out.append('</defs><g id="color-ir" data-mode="continuous-gradient-field">')
    for i, band in enumerate(rows):
        y=max(0,min(1000,int(band.get('y',0))))
        h=max(0,min(1000-y,int(band.get('height',0))))
        if h:
            # No stroked polygon boundaries, no per-cell tessellation.
            out.append(f'<rect x="0" y="{y}" width="1000" height="{h}" fill="url(#color-row-{i})"/>')
    out.append('</g><g id="geometry-ir">')
    for obj in scene['objects']:
        uid=escape(obj['id'],quote=True)
        geo=objects.get(obj['id'])
        out.append(f'<g data-object-id="{uid}" data-render-mode="'+('source-vector' if geo else 'color-field')+'">')
        if geo:
            for outline in geo.get('outline',[])[:20]:
                d=_vector_path(outline)
                if d:
                    out.append(path(d,stroke=obj['stroke'],sw=.9,extra='opacity="0.58"'))
            for part in geo.get('parts',[])[:130]:
                d=_vector_path(part.get('points',[]),closed=False)
                if d:
                    out.append(path(d,stroke=part.get('stroke',obj['stroke']),sw=.85,
                                    extra=f'data-part="{escape(part.get("id","detail"),quote=True)}" opacity="0.55"'))
        out.append('</g>')
    out.append('</g>')
    if details and details.get('mode')=='source_edge_paths':
        # The high-resolution colour field already contains original ink lines.
        # A strong full-detail overlay doubled the lines and made the image look
        # dark and dirty. Keep strokes independently editable but render gently.
        try:
            overlay_strength = float(os.getenv('DETAIL_OVERLAY_STRENGTH', '0.20'))
        except ValueError:
            overlay_strength = 0.20
        overlay_strength = max(0.0, min(1.0, overlay_strength))
        out.append(f'<g id="detail-ir" data-overlay-strength="{overlay_strength:.2f}" opacity="{overlay_strength:.2f}" fill="none" stroke-linecap="round" stroke-linejoin="round">')
        for stroke in details.get('strokes',[])[:12000]:
            d=_vector_path(stroke.get('points',[]),closed=False)
            if d:
                c=stroke.get('color','#343D3A')
                if not isinstance(c,str) or len(c)!=7 or not c.startswith('#'):
                    c='#343D3A'
                width=max(.3,min(2.5,float(stroke.get('width',.7))))
                opacity=max(.1,min(.9,float(stroke.get('opacity',.6))))
                out.append(f'<path d="{d}" stroke="{c}" stroke-width="{width:.2f}" opacity="{opacity:.2f}"/>')
        out.append('</g>')
    out.append('</svg>')
    return '\n'.join(out)

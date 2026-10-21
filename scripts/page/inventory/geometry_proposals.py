"""Traditional-vision proposals for nested visual components.

This module detects and records geometry only. It never alters source pixels;
layout export may turn accepted proposals into transparent child assets, and
the separate deferred-extraction gate controls whether a parent is patched.
"""

from __future__ import annotations

import cv2
import numpy as np


def _box(contour: np.ndarray, ox: int, oy: int) -> list[int]:
    x, y, w, h = cv2.boundingRect(contour)
    return [int(ox + x), int(oy + y), int(ox + x + w), int(oy + y + h)]


def _rectangularity(contour: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    return float(cv2.contourArea(contour)) / max(1.0, float(w * h))


def _evidence_family(method: str) -> str:
    """Collapse correlated channels before counting independent evidence."""
    if method.startswith('lab_edge_'):
        return 'edge'
    if method.startswith('pale_fill_'):
        return 'pale_fill'
    if method.startswith('hue_'):
        return 'hue'
    return method


def _repeated_rect_groups(candidates: list[dict]) -> list[dict]:
    """Group similarly sized rectangular proposals on aligned rows/columns."""
    rects = [c for c in candidates if c["kind"] == "rectangle"]
    groups: list[dict] = []
    used: set[int] = set()
    for i, seed in enumerate(rects):
        if i in used:
            continue
        x1, y1, x2, y2 = seed["bbox"]
        sw, sh = x2 - x1, y2 - y1
        members = [i]
        for j in range(i + 1, len(rects)):
            if j in used:
                continue
            bx1, by1, bx2, by2 = rects[j]["bbox"]
            bw, bh = bx2 - bx1, by2 - by1
            similar = (min(sw, bw) / max(sw, bw) >= 0.78
                       and min(sh, bh) / max(sh, bh) >= 0.72)
            aligned = (abs((y1 + y2) / 2 - (by1 + by2) / 2) <= 0.16 * max(sh, bh)
                       or abs((x1 + x2) / 2 - (bx1 + bx2) / 2) <= 0.16 * max(sw, bw))
            overlap = max(0,min(x2,bx2)-max(x1,bx1))*max(0,min(y2,by2)-max(y1,by1))
            if similar and aligned and overlap == 0:
                members.append(j)
        if len(members) < 3:
            continue
        group_boxes = [rects[k]["bbox"] for k in members]
        centers = np.array([[(b[0]+b[2])/2,(b[1]+b[3])/2] for b in group_boxes])
        axis = int(np.argmax(np.ptp(centers,axis=0)))
        transverse = 1-axis
        tolerance = .16 * (sh if axis == 0 else sw)
        if np.ptp(centers[:,transverse]) > tolerance:
            continue
        gaps = np.diff(np.sort(centers[:,axis]))
        if gaps.min() <= 0 or np.std(gaps)/np.mean(gaps) > .3:
            continue
        used.update(members)
        groups.append({
            "kind": "repeated_card_group",
            "member_bboxes": group_boxes,
            "bbox": [min(b[0] for b in group_boxes), min(b[1] for b in group_boxes),
                     max(b[2] for b in group_boxes), max(b[3] for b in group_boxes)],
            "member_count": len(group_boxes),
            "member_ids": [rects[k]['id'] for k in members],
            "axis": 'horizontal' if axis == 0 else 'vertical',
            "evidence": ["similar_size", "row_or_column_alignment"],
            "accepted_for_export": False,
            "decision_reason": "proposal_only_pending_parent_child_validation",
        })
    return groups


def propose_geometry(crop: np.ndarray, mask: np.ndarray,
                     origin: tuple[int, int] = (0, 0), min_area: int = 180,
                     ocr_boxes: list | None = None) -> dict:
    """Propose geometry and explain accepted and rejected evidence.

    The component mask is used as ownership evidence. Candidate contours may
    extend through the background inside a component bbox, but a proposal with
    no measurable support from its owning component is rejected. This avoids
    treating unrelated objects in a large bbox as children of that component.
    """
    diagnostics = {
        'contours_by_evidence': {},
        'contours_truncated_by_evidence': {},
        'rejected_by_reason': {},
        'rejected_samples': [],
        'rejected_samples_limit': 8,
        'duplicate_merges': 0,
        'hough_circle_count': 0,
        'candidate_limit': 180,
        'candidate_limit_reached': False,
        'component_mask_pixels': 0,
    }

    def reject(reason: str, method: str, bbox=None, area=None) -> None:
        diagnostics['rejected_by_reason'][reason] = (
            diagnostics['rejected_by_reason'].get(reason, 0) + 1)
        sample_reasons = {
            'too_small', 'covers_parent_bbox', 'unsupported_geometry',
            'high_ocr_overlap', 'outside_parent_component',
            'weak_parent_component_support',
        }
        if (reason in sample_reasons
                and len(diagnostics['rejected_samples'])
                < diagnostics['rejected_samples_limit']):
            sample = {'reason': reason, 'evidence': method}
            if bbox is not None:
                sample['bbox'] = [int(v) for v in bbox]
            if area is not None:
                sample['contour_area'] = round(float(area), 1)
            diagnostics['rejected_samples'].append(sample)

    if crop.size == 0:
        diagnostics['rejected_by_reason']['empty_crop'] = 1
        return {'candidates': [], 'groups': [], 'connections': [],
                'diagnostics': diagnostics}
    h, w = crop.shape[:2]
    mask_array = None if mask is None else np.asarray(mask)
    if mask_array is None or mask_array.ndim != 2 or mask_array.shape != (h, w):
        raise ValueError('component mask must match the geometry crop dimensions')
    component_mask = mask_array > 0
    diagnostics['component_mask_pixels'] = int(np.count_nonzero(component_mask))
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    evidence = []
    for channel in range(3):
        smoothed = cv2.GaussianBlur(lab[:, :, channel], (3, 3), .6)
        edges = cv2.Canny(smoothed, 8, 28)
        evidence.append((f'lab_edge_{channel}', edges))
    # Pale card interiors are visible as small colour offsets even where
    # their gradient border has no closed Canny contour. Restrict to light
    # fills, then remove narrow decorative bridges on detector copies only.
    for threshold in (5, 10):
        pale = ((np.min(crop,axis=2) > 205)
                & (255-np.min(crop,axis=2) > threshold)
                & (hsv[:,:,1] > 3)).astype(np.uint8)*255
        pale = cv2.morphologyEx(pale,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
        evidence.append((f'pale_fill_{threshold}',pale))
    # Hue bands are independent masks; joining them to the foreground would
    # swallow the diagonal boundary between adjacent workflow blocks.
    for hue in range(0, 180, 15):
        color = ((hsv[:, :, 0] >= hue) & (hsv[:, :, 0] < hue+15)
                 & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 45)).astype(np.uint8)*255
        evidence.append((f'hue_{hue}', color))
    candidates = []
    max_candidates = int(diagnostics['candidate_limit'])
    for method, binary in evidence:
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        diagnostics['contours_by_evidence'][method] = len(contours)
        ordered_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        diagnostics['contours_truncated_by_evidence'][method] = max(
            0, len(ordered_contours) - 500)
        for contour in ordered_contours[:500]:
            area = abs(float(cv2.contourArea(contour)))
            if area < max(min_area, 350 * (h/941)**2):
                reject('below_min_area', method, area=area)
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            if min(cw, ch) < 18 or cw*ch > .94*w*h:
                reason = 'too_small' if min(cw, ch) < 18 else 'covers_parent_bbox'
                reject(reason, method, [x+origin[0], y+origin[1],
                                        x+cw+origin[0], y+ch+origin[1]], area)
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                reject('degenerate_contour', method, area=area)
                continue
            approx = cv2.approxPolyDP(contour, max(.8, perimeter*.012), True)
            extent = area/max(1, cw*ch)
            circularity = 4*np.pi*area/(perimeter*perimeter)
            if circularity > .77 and .65 < cw/ch < 1.55 and len(approx) >= 6:
                kind = 'ellipse'
            elif extent > .86 and 4 <= len(approx) <= 12:
                kind = 'rectangle'
            elif 4 <= len(approx) <= 10 and extent > .48:
                kind = 'closed_polygon'
            else:
                reject('unsupported_geometry', method,
                       [x+origin[0], y+origin[1],
                        x+cw+origin[0], y+ch+origin[1]], area)
                continue
            bbox = _box(contour, *origin)
            bx1, by1, bx2, by2 = bbox
            text_fraction = 0.
            for tb in ocr_boxes or []:
                tx1, ty1, tx2, ty2 = tb
                overlap = max(0, min(bx2, tx2)-max(bx1, tx1))*max(0, min(by2, ty2)-max(by1, ty1))
                text_fraction += overlap / max(1, cw*ch)
            if text_fraction > .65:
                reject('high_ocr_overlap', method, bbox, area)
                continue

            # A filled-region fraction works for color masks; a thin contour
            # band also supports card outlines whose interiors are background.
            region = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(region, [contour], 1)
            boundary = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(boundary, [contour], -1, 1, thickness=3)
            region_pixels = int(np.count_nonzero(region))
            boundary_pixels = int(np.count_nonzero(boundary))
            region_support = int(np.count_nonzero((region > 0) & component_mask))
            boundary_support = int(np.count_nonzero((boundary > 0) & component_mask))
            component_support_pixels = int(np.count_nonzero(
                ((region > 0) | (boundary > 0)) & component_mask))
            region_fraction = region_support / max(1, region_pixels)
            boundary_fraction = boundary_support / max(1, boundary_pixels)
            support_fraction = max(region_fraction, boundary_fraction)
            min_boundary_support = max(3, int(round(perimeter * .02)))
            if (region_support == 0 and boundary_support == 0):
                reject('outside_parent_component', method, bbox, area)
                continue
            if (region_fraction < .01
                    and (boundary_support < min_boundary_support
                         or boundary_fraction < .04)):
                reject('weak_parent_component_support', method, bbox, area)
                continue
            duplicate = None
            for previous in candidates:
                ax1, ay1, ax2, ay2 = previous['bbox']
                inter = max(0, min(ax2,bx2)-max(ax1,bx1))*max(0, min(ay2,by2)-max(ay1,by1))
                union = (ax2-ax1)*(ay2-ay1)+cw*ch-inter
                if inter/max(1,union) > .85:
                    duplicate = previous
                    break
            if duplicate is not None:
                if method not in duplicate['evidence']:
                    duplicate['evidence'].append(method)
                family = _evidence_family(method)
                if family not in duplicate['evidence_families']:
                    duplicate['evidence_families'].append(family)
                duplicate['component_support_fraction'] = round(
                    max(duplicate['component_support_fraction'], support_fraction), 4)
                duplicate['component_region_support_fraction'] = round(
                    max(duplicate['component_region_support_fraction'], region_fraction), 4)
                duplicate['component_boundary_support_fraction'] = round(
                    max(duplicate['component_boundary_support_fraction'], boundary_fraction), 4)
                duplicate['component_support_pixels'] = max(
                    duplicate['component_support_pixels'],
                    component_support_pixels)
                diagnostics['duplicate_merges'] += 1
                continue
            candidates.append({
                'id': f'g{len(candidates):03d}', 'kind': kind, 'bbox': bbox,
                'vertices': len(approx), 'contour_area': round(area,1),
                'rectangularity': round(extent,4), 'circularity': round(float(circularity),4),
                'contour_points': [[int(p[0][0]+origin[0]), int(p[0][1]+origin[1])] for p in approx],
                'support_contours': [[[int(p[0][0]+origin[0]), int(p[0][1]+origin[1])]
                                      for p in contour]],
                'evidence': [method],
                'evidence_families': [_evidence_family(method)],
                'ocr_overlap_fraction': round(text_fraction,3),
                'component_support_fraction': round(support_fraction, 4),
                'component_region_support_fraction': round(region_fraction, 4),
                'component_boundary_support_fraction': round(boundary_fraction, 4),
                'component_support_pixels': component_support_pixels,
                'parent_id': None, 'accepted_for_export': False,
                'proposal_state': 'proposed',
                'decision_reason': 'awaiting_export_validation', 'exported_object_id': None,
            })
            if len(candidates) >= max_candidates:
                diagnostics['candidate_limit_reached'] = True
                break
        if len(candidates) >= max_candidates:
            diagnostics['candidate_limit_reached'] = True
            break
    # Interrupted luminous rings need a circle vote instead of a closed
    # contour. Extract the bright core icon separately from its rim/links;
    # the actual background gate still decides whether it can be moved.
    if (len(candidates) < max_candidates and min(h,w) >= 160
            and h*w >= 60000 and np.median(lab[:,:,0]) < 150):
        circles = cv2.HoughCircles(cv2.GaussianBlur(lab[:,:,0],(5,5),1),
                                  cv2.HOUGH_GRADIENT,1.2,35,param1=60,param2=28,
                                  minRadius=18,maxRadius=min(130,min(h,w)//3))
        diagnostics['hough_circle_count'] = 0 if circles is None else len(circles[0])
        centers = []
        edge_distance = cv2.distanceTransform(255-evidence[0][1],cv2.DIST_L2,3)
        for cx,cy,radius in ([] if circles is None else sorted(circles[0],key=lambda c:-c[2])[:40]):
            if len(candidates) >= max_candidates:
                diagnostics['candidate_limit_reached'] = True
                break
            angles = np.linspace(0,2*np.pi,96,endpoint=False)
            distances = []
            for offset in (-2,0,2):
                sx=np.rint(cx+(radius+offset)*np.cos(angles)).astype(int)
                sy=np.rint(cy+(radius+offset)*np.sin(angles)).astype(int)
                valid=(sx>=0)&(sy>=0)&(sx<w)&(sy<h)
                sampled=np.full(96,100.,dtype=float)
                sampled[valid]=edge_distance[sy[valid],sx[valid]]
                distances.append(sampled)
            if np.mean(np.min(distances,axis=0)<=2) < .70:
                reject('weak_ring_boundary', 'hough_ring', area=np.pi*radius*radius)
                continue
            if any(np.hypot(cx-px,cy-py)<.55*max(radius,pr) for px,py,pr in centers):
                reject('duplicate_hough_circle', 'hough_ring', area=np.pi*radius*radius)
                continue
            centers.append((cx,cy,radius))
            yy,xx=np.indices((h,w))
            core=(xx-cx)**2+(yy-cy)**2 < (.55*radius)**2
            values=hsv[:,:,2][core]
            if not len(values):
                reject('empty_circle_core', 'hough_ring')
                continue
            support=core & (hsv[:,:,2] > max(125,float(np.percentile(values,25))+40))
            component_support_pixels = int(np.count_nonzero(support & component_mask))
            if component_support_pixels < 4:
                reject('outside_parent_component', 'hough_ring')
                continue
            contours,_=cv2.findContours(support.astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours=[c for c in contours if cv2.contourArea(c)>6]
            if not contours or sum(cv2.contourArea(c) for c in contours)<40:
                reject('no_bright_core_contour', 'hough_ring')
                continue
            joined=np.concatenate(contours)
            bx,by,bw,bh=cv2.boundingRect(joined)
            bbox=[int(origin[0]+bx),int(origin[1]+by),int(origin[0]+bx+bw),int(origin[1]+by+bh)]
            if any(max(0,min(bbox[2],t[2])-max(bbox[0],t[0]))*
                   max(0,min(bbox[3],t[3])-max(bbox[1],t[1])) > .35*bw*bh for t in ocr_boxes or []):
                reject('high_ocr_overlap', 'hough_ring', bbox,
                       sum(cv2.contourArea(c) for c in contours))
                continue
            polygons=[[[int(p[0][0]+origin[0]),int(p[0][1]+origin[1])] for p in c] for c in contours]
            candidates.append({'id':f'g{len(candidates):03d}','kind':'node_icon','bbox':bbox,
                               'contour_points':polygons[0],'support_contours':polygons,
                               'contour_area':float(sum(cv2.contourArea(c) for c in contours)),
                               'circle':[float(cx+origin[0]),float(cy+origin[1]),float(radius)],
                               'evidence':['hough_ring','bright_core'],
                               'evidence_families':['hough_ring','bright_core'],
                               'component_support_fraction':round(
                                   component_support_pixels/max(1,int(np.count_nonzero(support))),4),
                               'component_region_support_fraction':round(
                                   component_support_pixels/max(1,int(np.count_nonzero(support))),4),
                               'component_boundary_support_fraction':0.0,
                               'component_support_pixels':component_support_pixels,
                               'parent_id':None,
                               'accepted_for_export':False,'exported_object_id':None,
                               'proposal_state':'proposed',
                               'decision_reason':'awaiting_export_validation'})
    # A geometric parent requires containment of the actual contour points,
    # rather than merely containment of the bounding rectangle.
    for child in candidates:
        parents = []
        for parent in candidates:
            if parent['contour_area'] <= child['contour_area'] * 1.2:
                continue
            poly = np.array(parent['contour_points'], dtype=np.float32)
            if all(cv2.pointPolygonTest(poly, tuple(map(float, p)), False) >= 0
                   for p in child['contour_points']):
                parents.append(parent)
        if parents:
            child['parent_id'] = min(parents, key=lambda p:p['contour_area'])['id']
    groups = _repeated_rect_groups(candidates)
    # Node/edge topology is diagnostic until background reconstruction can
    # preserve the glow and any crossing lines. No line is erased here.
    node_edges = evidence[0][1].copy()
    nodes = [c for c in candidates if c['kind'] == 'ellipse']
    for node in nodes:
        pts = np.asarray(node['contour_points'], np.int32)-np.asarray(origin, np.int32)
        cv2.fillPoly(node_edges, [pts], 0)
    lines = cv2.HoughLinesP(node_edges, 1, np.pi/180, 28, minLineLength=30, maxLineGap=7)
    connections = []
    for segment in ([] if lines is None else lines[:160]):
        x1,y1,x2,y2 = map(int,segment[0])
        endpoints = [[x1+origin[0],y1+origin[1]],[x2+origin[0],y2+origin[1]]]
        linked = []
        for point in endpoints:
            matches = [(abs(cv2.pointPolygonTest(np.asarray(n['contour_points'],np.float32),
                                                 tuple(map(float, point)),True)), n['id']) for n in nodes]
            linked.append(min(matches)[1] if matches and min(matches)[0] < 15 else None)
        if all(linked) and linked[0] != linked[1]:
            connections.append({'endpoints': endpoints, 'node_ids': linked,
                                'state': 'proposed', 'reason': 'line_reconstruction_not_certified'})
    return {'candidates': candidates, 'groups': groups, 'connections': connections,
            'diagnostics': diagnostics}



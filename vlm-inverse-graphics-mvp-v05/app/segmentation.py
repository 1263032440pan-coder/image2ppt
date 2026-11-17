"""Pluggable semantic mask handling and conservative vector-mask rasterization.

No SAM weights shipped. OpenCV GrabCut is the working MVP adapter. External
SAM integration can replace extract_geometry without altering Color/Detail IR.
"""
from __future__ import annotations
import cv2
import numpy as np


def mask_from_outline(outlines: list, size: int = 256) -> np.ndarray:
    mask=np.zeros((size,size),dtype=np.uint8)
    for poly in outlines[:32]:
        if not isinstance(poly,list) or len(poly)<3:
            continue
        pts=np.asarray(poly,dtype=np.float32)
        if not np.isfinite(pts).all():
            continue
        pts=np.clip(pts/1000*size,0,size-1).astype(np.int32)
        if len(pts)>2:
            cv2.fillPoly(mask,[pts],255)
    return mask


def visible_instance_masks(scene: dict,geometry: dict,size:int=256) -> dict[str,np.ndarray]:
    """Occluder-aware *estimated* visible masks (only reliably traced objects).

    Overlap pixels belonging to a higher foreground object are excluded. Lack
    of a source mask is not treated as evidence that an object does not exist.
    """
    out={}
    occupied=np.zeros((size,size),dtype=np.uint8)
    geo=(geometry or {}).get('objects',{})
    for obj in sorted(scene.get('objects',[]),key=lambda o:o['layer'],reverse=True):
        g=geo.get(obj['id'])
        if g and g.get('outline'):
            own=mask_from_outline(g['outline'],size)
            visible=cv2.bitwise_and(own,cv2.bitwise_not(occupied))
            out[obj['id']]=visible
            occupied=cv2.bitwise_or(occupied,own)
    return out

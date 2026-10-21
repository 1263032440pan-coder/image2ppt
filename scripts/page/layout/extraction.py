"""Reversible, pixel-mask extraction after assets have actually been emitted."""
from __future__ import annotations

import cv2
import numpy as np


def fit_background(image: np.ndarray, mask: np.ndarray):
    """Fit an affine colour surface to a held-out local ring.

    A busy ring (including crossing connectors) fails closed. The fitted
    surface is only a reconstruction hypothesis, never inferred from the
    foreground pixels being removed. Return a separate image, not a mutation.
    """
    mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if len(xs) < 4:
        return None, {"reason": "empty_mask"}
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    # Sample the immediate background, not a band proportional to the
    # object's size: a large card's wide band would include adjacent cards
    # or its container's header and falsely declare the background busy.
    pad = max(3, min(8, int(round(image.shape[0] / 941 * 4))))
    if x1 < pad or y1 < pad or x2 + pad > image.shape[1] or y2 + pad > image.shape[0]:
        return None, {"reason": "insufficient_background_ring"}
    local = image[y1-pad:y2+pad, x1-pad:x2+pad, :3]
    support = mask[y1-pad:y2+pad, x1-pad:x2+pad]
    ring = cv2.dilate(support.astype(np.uint8), np.ones((2*pad+1, 2*pad+1), np.uint8)) > 0
    ring &= ~support
    yy, xx = np.indices(support.shape)
    design = np.stack([np.ones_like(xx), xx / max(1, support.shape[1]),
                       yy / max(1, support.shape[0])], axis=-1)
    train = ring & ((xx + yy) % 2 == 0)
    check = ring & ~train
    if min(int(train.sum()), int(check.sum())) < 24:
        return None, {"reason": "insufficient_background_samples"}
    coef, _, _, _ = np.linalg.lstsq(design[train], local[train].astype(float), rcond=None)
    predicted = design @ coef
    error = np.max(np.abs(predicted[check] - local[check]), axis=1)
    metrics = {"ring_p95_error": round(float(np.percentile(error, 95)), 3),
               "ring_outlier_fraction": round(float(np.mean(error > 24)), 5)}
    if metrics["ring_p95_error"] > 12 or metrics["ring_outlier_fraction"] > .01:
        return None, {"reason": "textured_background_or_crossing_structure", **metrics}
    out = image.copy()
    patch = out[y1-pad:y2+pad, x1-pad:x2+pad, :3]
    patch[support] = np.clip(np.rint(predicted[support]), 0, 255).astype(np.uint8)
    return out, {"reason": "smooth_background_validated", **metrics}


def validate_child_asset(asset: np.ndarray | None, expected: np.ndarray,
                         removal_mask: np.ndarray) -> str | None:
    """Require real, opaque, pixel-faithful replacement of every erased pixel."""
    if asset is None or asset.shape[:2] != expected.shape[:2]:
        return "missing_or_wrong_size_asset"
    if asset.ndim != 3 or asset.shape[2] not in (3, 4):
        return "invalid_asset_channels"
    if asset.shape[2] == 4 and np.any(asset[:, :, 3][removal_mask] < 254):
        return "replacement_not_opaque"
    diff = np.abs(asset[:, :, :3].astype(np.int16) - expected[:, :, :3].astype(np.int16))
    if np.any(diff[removal_mask] > 2):
        return "replacement_pixels_differ"
    return None

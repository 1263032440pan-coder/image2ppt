"""InventoryBuilder: orchestrates cleaned image + OCR → inventory.json.

Lifts the ~20 previously-nested closures inside build_inventory._run
into instance methods on a single class, so each step can be navigated
and tested independently. Runtime behaviour is unchanged — this is a
pure refactor of where the code lives.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PAGE_DIR = Path(__file__).resolve().parents[1]    # scripts/page
SCRIPTS_ROOT = PAGE_DIR.parent                     # scripts
for _p in (PAGE_DIR, SCRIPTS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.geometry import connector_on_container_border  # noqa: E402
from erase_text import (  # noqa: E402
    detect_logo_strips,
    is_in_logo_zone,
    is_likely_icon,
    preprocess_ocr,
    should_preserve_visual,
)
from _heuristics import pixel_scale, s_area, s_kernel, s_length  # noqa: E402
from icon import (  # noqa: E402
    detect_internal_shapes,
    detect_line_art_subicons,
    detect_white_subicons,
    inpaint_region_inplace,
)

from inventory.badge_trim import trim_filled_badge  # noqa: E402
from inventory.child_fill import (  # noqa: E402
    clean_child_fill,
    ring_bg_quality,
    simulated_fill_quality,
)
from inventory.role_classify import (  # noqa: E402
    foreground_role_for_box,
    is_connector_like,
    is_container_like,
)
from inventory.geometry_proposals import propose_geometry  # noqa: E402


def _box4(record: tuple) -> tuple[int, int, int, int]:
    return (int(record[0]), int(record[1]),
            int(record[2]), int(record[3]))


def _inventory_sort_key(e: dict) -> tuple:
    """Stable order: image first, by role priority, then top-to-bottom."""
    role_order = {
        "background": 0, "container": 1, "outline": 2, "internal": 3,
        "badge_subicon": 4, "connector": 4, "line_subicon": 4, "subicon": 4,
    }
    if e.get("type") == "image":
        role = e.get("role")
        return (0, role_order.get(role, 1), e["bbox"][1], e["bbox"][0])
    return (1, 0, e["bbox"][1], e["bbox"][0])


class InventoryBuilder:
    """Drive cleaned + OCR → inventory.json + per-component masks.

    Owns the per-page state the previous closure-heavy `_run` function
    carried implicitly: source/cleaned/icon_probe images, OCR splits,
    scaling factors, and the six record buckets (background, outline,
    foreground, subicon, line_subicon, internal_shape).
    """

    def __init__(self, args):
        from build_inventory import (
            detect_components,
            detect_outline_mask,
            detect_outline_rects,
            conservative_split,
            _foreground_mask,
        )
        self._detect_components = detect_components
        self._detect_outline_mask = detect_outline_mask
        self._detect_outline_rects = detect_outline_rects
        self._conservative_split = conservative_split
        self._foreground_mask = _foreground_mask

        self.args = args
        self.cleaned = cv2.imread(args.clean)
        self.source = cv2.imread(args.source)
        if self.cleaned is None or self.source is None:
            raise SystemExit("Could not load images.")
        self.ocr_data = preprocess_ocr(
            json.loads(Path(args.ocr).read_text(encoding="utf-8")),
            img=self.source,
        )
        # Icon detectors want only real text glyphs, not OCR-misclassified
        # visuals like numbered badges read as "②".
        self.ocr_text_items = [
            it for it in self.ocr_data
            if not should_preserve_visual(it)
        ]

        # Every pixel constant in this pipeline was tuned at 720-tall source.
        # Silently scaling user-facing knobs keeps default-flow runs correct.
        self.scale = pixel_scale(self.cleaned)
        args.min_area = s_area(args.min_area, self.scale)
        args.dilate = s_kernel(args.dilate, self.scale)
        args.split_gap = s_length(args.split_gap, self.scale)

        # Save a sidecar TEXT-ONLY-erased cleaned image BEFORE we modify
        # `cleaned` in place. Downstream asset cropping uses this sidecar.
        clean_path = Path(args.clean)
        self.text_only_path = clean_path.with_name(
            f"{clean_path.stem}.text_only.png")
        cv2.imwrite(str(self.text_only_path), self.cleaned)
        # Icon detection must run on the text-erased view, not the source.
        self.icon_probe = self.cleaned.copy()

        # Pre-filter OCR for likely-icons / logo-band texts.
        logo_bands = detect_logo_strips(self.ocr_data, self.cleaned.shape[0])
        self.candidate_texts = [
            item for item in self.ocr_data
            if not is_likely_icon(item, self.ocr_data, self.source)[0]
            and not is_in_logo_zone(item, logo_bands)
        ]

        # Compute one shared foreground mask. The connected-component label
        # map below lets each split pass use only the pixels belonging to
        # that component, avoiding projection contamination from neighbors.
        self.component_mask = self._foreground_mask(
            self.cleaned, args.dilate)
        (self.components, self.component_label_map,
         self.component_label_ids) = self._detect_components(
            self.cleaned, args.min_area, args.dilate,
            foreground_mask=self.component_mask, return_labels=True)
        self.component_split_records: list[dict] = []
        self.geometry_proposal_records: list[dict] = []
        self.extraction_masks: dict[tuple, tuple] = {}
        self.composite_records: list[dict] = []
        self.foreground_parent_keys: dict[tuple[int, int, int, int], str] = {}
        self.composite_output_ids: dict[str, str] = {}

        self.inventory: list[dict] = []
        self.visual_idx = 0
        self.img_h, self.img_w = self.cleaned.shape[:2]
        self.background_records: list[tuple] = []
        self.outline_records: list[tuple] = []
        self.foreground_records: list[tuple] = []
        self.subicon_records: list[tuple] = []
        self.line_subicon_records: list[tuple] = []
        self.internal_shape_records: list[tuple] = []
        self.inpainted_children: set[tuple[int, int, int, int]] = set()
        self.unclean_nested_children: set[tuple[int, int, int, int]] = set()

        # Pre-compute scaled gates used by the scan helpers.
        self.parent_min_side = s_length(120, self.scale)
        self.line_min_area = s_area(1200, self.scale)
        self.line_min_side = s_length(24, self.scale)

    # ------------------------------------------------------------------
    # Record-append helpers
    # ------------------------------------------------------------------

    def _append_foreground_record(self, x1: int, y1: int,
                                  x2: int, y2: int,
                                  *, force: bool = False) -> None:
        """Append a foreground bbox unless an equivalent one exists."""
        area = max(1, (x2 - x1) * (y2 - y1))
        for existing in self.foreground_records:
            ex1, ey1, ex2, ey2 = _box4(existing)
            earea = max(1, (ex2 - ex1) * (ey2 - ey1))
            size_ratio = min(area, earea) / max(area, earea)
            if force or size_ratio < 0.82:
                continue
            ix1, iy1 = max(x1, ex1), max(y1, ey1)
            ix2, iy2 = min(x2, ex2), min(y2, ey2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            if (ix2 - ix1) * (iy2 - iy1) >= 0.92 * min(area, earea):
                return
        self.foreground_records.append(
            (int(x1), int(y1), int(x2), int(y2)))

    def _append_outline_record(self, x1: int, y1: int, x2: int, y2: int,
                               outline_mask: np.ndarray) -> None:
        """Dedup against existing outlines before appending."""
        area = max(1, (x2 - x1) * (y2 - y1))
        for ox1, oy1, ox2, oy2, _om in self.outline_records:
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
            similar_size = min(area, oarea) / max(area, oarea) >= 0.55
            if similar_size and (ix2 - ix1) * (iy2 - iy1) >= 0.85 * min(area, oarea):
                return
        self.outline_records.append((x1, y1, x2, y2, outline_mask))

    # ------------------------------------------------------------------
    # Sub-shape scans
    # ------------------------------------------------------------------

    def _remember_extraction(self, box, parent_box, mask) -> None:
        """Keep detector support; never modify pixels before child export."""
        x1, y1, x2, y2 = (int(v) for v in box)
        px1, py1, _, _ = (int(v) for v in parent_box)
        support = mask[y1-py1:y2-py1, x1-px1:x2-px1].copy()
        if support.shape == (y2-y1, x2-x1) and np.count_nonzero(support) >= 4:
            self.extraction_masks[(x1, y1, x2, y2)] = (
                support.astype(bool), list(parent_box))

    def _scan_internal_shapes_inplace(self, sx1: int, sy1: int,
                                      sx2: int, sy2: int) -> None:
        """Detect internal sub-shapes inside a parent component."""
        if (sx2 - sx1 < self.parent_min_side
                or sy2 - sy1 < self.parent_min_side):
            return
        shapes, fill_jobs = detect_internal_shapes(
            self.icon_probe, sx1, sy1, sx2, sy2,
            ocr_text_items=self.ocr_text_items,
            min_dim=s_length(20, self.scale),
            max_dim=s_length(220, self.scale),
            min_area=s_area(400, self.scale),
            scale=self.scale,
        )
        if not shapes:
            return
        source_local = self.icon_probe[sy1:sy2, sx1:sx2]
        for (ix1, iy1, ix2, iy2), (mask_in_crop, color) in zip(shapes, fill_jobs):
            ok, fill = clean_child_fill(source_local, mask_in_crop, self.scale, color)
            if not ok:
                continue
            if self._shape_box_already_covered(ix1, iy1, ix2, iy2):
                continue
            self.internal_shape_records.append((ix1, iy1, ix2, iy2))
            self._remember_extraction((ix1, iy1, ix2, iy2),
                                      (sx1, sy1, sx2, sy2), mask_in_crop)

    def _split_component(self, crop: np.ndarray,
                         component_mask: np.ndarray,
                         origin: tuple[int, int] = (0, 0)
                         ) -> tuple[list[tuple], str, dict]:
        """Recursively split a connected component using geometric evidence.

        Candidate boundaries come from the existing whitespace projections
        and from several scale-adjusted morphological openings. Each split is
        scored for foreground coverage and child balance before recursion.
        The original pixels are never changed here; this stage only builds
        boxes and an auditable decision tree.
        """
        page_area = max(1, self.img_w * self.img_h)
        max_depth = 3
        max_nodes = 128
        pad = s_length(3, self.scale)
        selected_methods: list[str] = []
        nodes: list[dict] = []

        def page_box(local_box, abs_x: int, abs_y: int) -> list[int]:
            bx1, by1, bx2, by2 = local_box
            return [int(abs_x + bx1), int(abs_y + by1),
                    int(abs_x + bx2), int(abs_y + by2)]

        def split_quality(mask: np.ndarray, boxes: list[tuple],
                          depth: int, opened_candidate: bool
                          ) -> tuple[float, float, str]:
            h, w = mask.shape[:2]
            total = int(np.count_nonzero(mask))
            if len(boxes) <= 1:
                return 0.0, 0.0, "no_geometric_gap"
            if len(boxes) > 32:
                return 0.0, 0.0, "too_many_children"
            min_side = s_length(20 if depth == 0 else 24, self.scale)
            min_area = s_area(450 if depth == 0 else 600, self.scale)
            covered = np.zeros((h, w), dtype=np.uint8)
            child_pixels: list[int] = []
            for bx1, by1, bx2, by2 in boxes:
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)
                if bx2 <= bx1 or by2 <= by1:
                    continue
                bw, bh = bx2 - bx1, by2 - by1
                if (opened_candidate or depth > 0) and (
                    min(bw, bh) < min_side or bw * bh < min_area
                ):
                    return 0.0, 0.0, "child_below_minimum_size"
                covered[by1:by2, bx1:bx2] = 1
                child_pixels.append(int(np.count_nonzero(
                    mask[by1:by2, bx1:bx2])))
            coverage = (
                float(np.count_nonzero((mask > 0) & (covered > 0)))
                / max(1, total)
            )
            if coverage < 0.78:
                return coverage, 0.0, "low_foreground_coverage"
            nonzero_children = [count for count in child_pixels if count > 0]
            if len(nonzero_children) != len(boxes):
                return coverage, 0.0, "empty_child_region"
            mean_child = sum(nonzero_children) / max(1, len(nonzero_children))
            balance = min(nonzero_children) / max(1.0, mean_child)
            # Coverage dominates to avoid losing faint borders; balance
            # favours useful panels over one large panel plus tiny fragments.
            score = 0.78 * coverage + 0.22 * balance
            if len(boxes) > 10:
                score -= min(0.25, 0.025 * (len(boxes) - 10))
            return coverage, score, "accepted"

        def color_seam_boxes(node_crop: np.ndarray, axis: int) -> list[tuple]:
            """Find broad colour transitions crossing a panel, not text edges."""
            nh, nw = node_crop.shape[:2]
            length = nh if axis == 0 else nw
            if length < s_length(100, self.scale):
                return []
            lab = cv2.cvtColor(node_crop, cv2.COLOR_BGR2LAB).astype(np.int16)
            delta = np.max(np.abs(np.diff(lab, axis=axis)), axis=2)
            fractions = np.mean(delta >= 18, axis=1 if axis == 0 else 0)
            edge = max(s_length(22, self.scale), int(length * 0.07))
            separation = max(s_length(36, self.scale), int(length * 0.12))
            ranked = np.argsort(fractions)[::-1]
            cuts: list[int] = []
            for index in ranked:
                cut = int(index) + 1
                if fractions[index] < 0.48:
                    break
                if cut < edge or length - cut < edge:
                    continue
                if any(abs(cut - prior) < separation for prior in cuts):
                    continue
                cuts.append(cut)
                if len(cuts) >= 3:
                    break
            if not cuts:
                return []
            edges = [0] + sorted(cuts) + [length]
            if axis == 0:
                return [(0, a, nw, b) for a, b in zip(edges, edges[1:])]
            return [(a, 0, b, nh) for a, b in zip(edges, edges[1:])]

        def repeated_text_row_boxes(node_crop: np.ndarray,
                                    abs_x: int, abs_y: int) -> list[tuple]:
            """Partition aligned light-card lists at gaps between OCR rows."""
            nh, nw = node_crop.shape[:2]
            if (nw > 0.29 * self.img_w
                    or nh < s_length(180, self.scale)
                    or float(np.median(cv2.cvtColor(
                        node_crop, cv2.COLOR_BGR2GRAY))) < 175):
                return []
            intervals = []
            for item in self.ocr_text_items:
                tx1, ty1 = int(item["x1"]), int(item["y1"])
                tx2, ty2 = int(item["x2"]), int(item["y2"])
                if (tx1 < abs_x or tx2 > abs_x + nw
                        or ty1 < abs_y or ty2 > abs_y + nh
                        or tx2 - tx1 < s_length(15, self.scale)
                        or ty2 - ty1 < s_length(8, self.scale)):
                    continue
                intervals.append((ty1 - abs_y, ty2 - abs_y))
            intervals.sort()
            if len(intervals) < 6:
                return []
            merge_gap = s_length(24, self.scale)
            groups: list[list[int]] = []
            for start, end in intervals:
                if groups and start - groups[-1][1] <= merge_gap:
                    groups[-1][1] = max(groups[-1][1], end)
                else:
                    groups.append([start, end])
            if not 3 <= len(groups) <= 8:
                return []
            centers = np.array([(a + b) / 2 for a, b in groups])
            spacing = np.diff(centers)
            if (float(np.min(spacing)) < s_length(45, self.scale)
                    or float(np.std(spacing) / np.mean(spacing)) > 0.25):
                return []
            cuts = [int((left[1] + right[0]) / 2)
                    for left, right in zip(groups, groups[1:])]
            edges = [0] + cuts + [nh]
            if any(b - a < s_length(42, self.scale)
                   for a, b in zip(edges, edges[1:])):
                return []
            return [(0, a, nw, b) for a, b in zip(edges, edges[1:])]

        def visit(node_crop: np.ndarray, node_mask: np.ndarray, depth: int,
                  abs_x: int, abs_y: int,
                  exact_only: bool = False) -> list[tuple]:
            h, w = node_mask.shape[:2]
            box = (0, 0, w, h)
            if len(nodes) >= max_nodes:
                return [box]
            node = {
                "bbox": page_box(box, abs_x, abs_y),
                "depth": int(depth),
                "bbox_page_fraction": round((w * h) / page_area, 5),
                "candidates": [],
                "children": [],
                "selected_method": None,
                "decision_reason": None,
            }
            nodes.append(node)

            if depth >= max_depth:
                node["decision_reason"] = "max_depth"
                return [box]

            candidates: list[tuple[str, int | None, list[tuple], bool]] = []
            if not exact_only:
                projection = self._conservative_split(
                    node_crop, min_gap=self.args.split_gap,
                    foreground_mask=node_mask)
                candidates.append(("projection", None, projection, False))
            node_page_fraction = (w * h) / page_area
            row_boxes = repeated_text_row_boxes(node_crop, abs_x, abs_y)
            if row_boxes:
                candidates.append(("text_rows_horizontal", None,
                                   row_boxes, False))
            if node_page_fraction >= 0.018:
                for axis, direction in ((0, "horizontal"),
                                        (1, "vertical")):
                    seam_boxes = color_seam_boxes(node_crop, axis)
                    if seam_boxes:
                        candidates.append((
                            f"color_seam_{direction}", None,
                            seam_boxes, False))

            # Try a small family of openings. The winning kernel is selected
            # from observed coverage/fragment balance instead of one fixed
            # kernel for every component size.
            min_open_fraction = 0.04 if depth == 0 else 0.018
            if (not exact_only and depth < max_depth
                    and node_page_fraction >= min_open_fraction
                    and min(h, w) >= s_length(70, self.scale)):
                max_kernel = max(3, min(h, w) // 8)
                if max_kernel % 2 == 0:
                    max_kernel -= 1
                for nominal in (5, 7, 9, 13):
                    kernel_size = max(3, s_length(nominal, self.scale))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    kernel_size = min(kernel_size, max_kernel)
                    if kernel_size < 3:
                        continue
                    if any(c[1] == kernel_size for c in candidates):
                        continue
                    opened = cv2.morphologyEx(
                        node_mask, cv2.MORPH_OPEN,
                        cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (kernel_size, kernel_size)))
                    if int(np.count_nonzero(opened)) < s_area(180, self.scale):
                        continue
                    opened_boxes = self._conservative_split(
                        node_crop, min_gap=self.args.split_gap,
                        foreground_mask=opened)
                    candidates.append((
                        "morph_open+projection", kernel_size,
                        opened_boxes, True))
                    component_count, _, component_stats, _ = (
                        cv2.connectedComponentsWithStats(opened, 8))
                    component_boxes = []
                    for label in range(1, component_count):
                        bx, by, bw, bh, area = component_stats[label]
                        if area < s_area(180, self.scale):
                            continue
                        if min(bw, bh) < s_length(14, self.scale):
                            continue
                        component_boxes.append((
                            int(bx), int(by), int(bx + bw), int(by + bh)))
                    candidates.append((
                        "morph_open+components", kernel_size,
                        component_boxes or [(0, 0, w, h)], True))

            best = None
            for method_name, kernel_size, boxes, opened_candidate in candidates:
                coverage, score, reason = split_quality(
                    node_mask, boxes, depth, opened_candidate)
                if reason == "accepted" and not (
                    method_name.startswith("color_seam")
                    or method_name.startswith("text_rows")):
                    areas = sorted(
                        ((bx2 - bx1) * (by2 - by1)
                         for bx1, by1, bx2, by2 in boxes), reverse=True)
                    node_area = max(1, w * h)
                    if (areas[0] >= 0.80 * node_area
                            and sum(areas[1:]) < 0.12 * node_area):
                        reason = "dominant_core_with_fragments"
                        score = 0.0
                if len(boxes) > 1 and reason == "accepted":
                    # A projection-only split at the root remains compatible
                    # with the old thresholds; recursive splits are stricter.
                    candidate_score = score
                    if depth == 0 and method_name == "projection":
                        candidate_score += 0.005
                    if method_name.startswith("text_rows"):
                        # A regularly spaced list of OCR rows gives more
                        # useful card objects than a header-only seam.
                        candidate_score += 0.15
                    if best is None or candidate_score > best["score"]:
                        best = {
                            "method": method_name,
                            "kernel_size": kernel_size,
                            "boxes": boxes,
                            "score": candidate_score,
                            "coverage": coverage,
                        }
                node["candidates"].append({
                    "method": method_name,
                    "kernel_size": kernel_size,
                    "split_count": int(len(boxes)),
                    "coverage": round(coverage, 4),
                    "score": round(score, 4),
                    "accepted": bool(len(boxes) > 1 and reason == "accepted"),
                    "reason": reason,
                    "split_bboxes": [page_box(b, abs_x, abs_y)
                                     for b in boxes],
                })

            for entry in node["candidates"]:
                entry["selected"] = bool(
                    best is not None
                    and entry["method"] == best["method"]
                    and entry["kernel_size"] == best["kernel_size"])

            if best is None:
                reasons = [entry["reason"] for entry in node["candidates"]]
                node["decision_reason"] = (
                    "no_geometric_gap" if all(
                        reason == "no_geometric_gap" for reason in reasons
                    ) else "all_candidates_rejected")
                return [box]

            node["selected_method"] = best["method"]
            node["decision_reason"] = "split"
            selected_methods.append(
                best["method"] + (f"({best['kernel_size']})"
                                   if best["kernel_size"] else ""))
            if (best["method"].startswith("color_seam")
                    or best["method"].startswith("text_rows")):
                # Exact partitions may recurse, but must never be padded.
                output: list[tuple] = []
                for bx1, by1, bx2, by2 in best["boxes"]:
                    child_crop = node_crop[by1:by2, bx1:bx2]
                    child_mask = node_mask[by1:by2, bx1:bx2]
                    child_offset = (abs_x + bx1, abs_y + by1)
                    if best["method"].startswith("text_rows"):
                        child_leaves = [(0, 0, bx2 - bx1, by2 - by1)]
                        child_node_index = None
                    else:
                        child_node_index = len(nodes)
                        child_leaves = visit(
                            child_crop, child_mask, depth + 1,
                            *child_offset, exact_only=True)
                    node["children"].append({
                        "bbox": page_box((bx1, by1, bx2, by2),
                                         abs_x, abs_y),
                        "node_index": child_node_index,
                        "decision_reason": "exact_partition",
                    })
                    output.extend((
                        (int(bx1 + sx1), int(by1 + sy1),
                         int(bx1 + sx2), int(by1 + sy2))
                        for sx1, sy1, sx2, sy2 in child_leaves
                    ))
                return output
            output: list[tuple] = []
            for bx1, by1, bx2, by2 in best["boxes"]:
                px1, py1 = max(0, bx1 - pad), max(0, by1 - pad)
                px2, py2 = min(w, bx2 + pad), min(h, by2 + pad)
                if px2 <= px1 or py2 <= py1:
                    continue
                child_mask = node_mask[py1:py2, px1:px2].copy()
                child_crop = node_crop[py1:py2, px1:px2]
                child_offset = (abs_x + px1, abs_y + py1)
                before = len(nodes)
                child_leaves = visit(child_crop, child_mask, depth + 1,
                                     *child_offset)
                node["children"].append({
                    "bbox": [int(child_offset[0]), int(child_offset[1]),
                             int(child_offset[0] + child_crop.shape[1]),
                             int(child_offset[1] + child_crop.shape[0])],
                    "node_index": (int(before) if len(nodes) > before else None),
                    "decision_reason": (
                        None if len(nodes) > before else "node_budget"),
                })
                output.extend((
                    (int(px1 + sx1), int(py1 + sy1),
                     int(px1 + sx2), int(py1 + sy2))
                    for sx1, sy1, sx2, sy2 in child_leaves
                ))
            return output or [box]

        leaves = visit(crop, component_mask, 0, int(origin[0]), int(origin[1]))
        tree = {
            "root_bbox": [int(origin[0]), int(origin[1]),
                          int(origin[0] + crop.shape[1]),
                          int(origin[1] + crop.shape[0])],
            "max_depth": max_depth,
            "max_nodes": max_nodes,
            "nodes": nodes,
            "leaf_bboxes": [page_box(box, *origin) for box in leaves],
        }
        # Remove adjacent duplicate method names while preserving order.
        method_chain = list(dict.fromkeys(selected_methods))
        method_summary = ">".join(method_chain) if method_chain else "projection"
        tree["split_method"] = method_summary
        tree["unsplit_reason"] = (
            nodes[0].get("decision_reason") if len(leaves) == 1 else None)
        return leaves, method_summary, tree

    def _scan_subicons_inplace(self, sx1: int, sy1: int,
                               sx2: int, sy2: int) -> None:
        """Detect white pictograms inside dark uniform sub-shapes."""
        subs, fill_jobs = detect_white_subicons(
            self.icon_probe, sx1, sy1, sx2, sy2,
            ocr_text_items=self.ocr_text_items,
            min_dim=s_length(15, self.scale),
            max_dim=s_length(220, self.scale),
            min_area=s_area(200, self.scale),
            scale=self.scale,
        )
        if not subs:
            return
        source_local = self.icon_probe[sy1:sy2, sx1:sx2]
        for (ix1, iy1, ix2, iy2), (mask_in_crop, color) in zip(subs, fill_jobs):
            ok, fill = clean_child_fill(source_local, mask_in_crop, self.scale, color)
            if not ok:
                continue
            if any(self._box_overlap_ratio(
                    (ix1, iy1, ix2, iy2), _box4(r)) >= 0.5
                   for r in self.subicon_records):
                continue
            color_lum = float(
                0.114 * int(fill[0])
                + 0.587 * int(fill[1])
                + 0.299 * int(fill[2])
            )
            role = "badge_subicon" if color_lum > 160 else "subicon"
            self.subicon_records.append((ix1, iy1, ix2, iy2, role))
            self._remember_extraction((ix1, iy1, ix2, iy2),
                                      (sx1, sy1, sx2, sy2), mask_in_crop)

    def _scan_line_subicons_inplace(self, sx1: int, sy1: int,
                                    sx2: int, sy2: int) -> None:
        """Detect sparse line-art children inside this parent bbox."""
        sw = sx2 - sx1
        sh = sy2 - sy1
        if sw * sh < self.line_min_area or min(sw, sh) < self.line_min_side:
            return
        subs, fill_jobs = detect_line_art_subicons(
            self.icon_probe, sx1, sy1, sx2, sy2,
            ocr_text_items=self.ocr_text_items,
            min_dim=s_length(12, self.scale),
            max_dim=s_length(120, self.scale),
            min_area=s_area(35, self.scale),
            scale=self.scale,
        )
        if not subs:
            return
        for (ix1, iy1, ix2, iy2), (mask_in_crop, _color) in zip(subs, fill_jobs):
            if self._line_subicon_carried_by_container(ix1, iy1, ix2, iy2):
                continue
            source_local = self.icon_probe[sy1:sy2, sx1:sx2]
            ix1, iy1, ix2, iy2, child_mask = trim_filled_badge(self.icon_probe, self.ocr_text_items, self.scale, 
                ix1, iy1, ix2, iy2, mask_in_crop, (sx1, sy1))
            ok, fill = clean_child_fill(source_local, child_mask, self.scale, _color)
            if not ok:
                continue
            if self._shape_box_already_covered(ix1, iy1, ix2, iy2):
                continue
            full_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            full_mask[sy1:sy2, sx1:sx2] = child_mask.astype(np.uint8) * 255
            self.line_subicon_records.append((ix1, iy1, ix2, iy2, full_mask))
            self._remember_extraction((ix1, iy1, ix2, iy2),
                                      (sx1, sy1, sx2, sy2), child_mask)

    def _shape_box_already_covered(self, ix1: int, iy1: int,
                                   ix2: int, iy2: int) -> bool:
        existing = (
            [_box4(r) for r in self.subicon_records]
            + [_box4(r) for r in self.line_subicon_records]
            + [_box4(r) for r in self.internal_shape_records]
        )
        for ex1, ey1, ex2, ey2 in existing:
            if self._box_overlap_ratio(
                    (ix1, iy1, ix2, iy2), (ex1, ey1, ex2, ey2)) >= 0.5:
                return True
        return False

    @staticmethod
    def _box_overlap_ratio(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ox1, oy1 = max(ax1, bx1), max(ay1, by1)
        ox2, oy2 = min(ax2, bx2), min(ay2, by2)
        if ox2 <= ox1 or oy2 <= oy1:
            return 0.0
        inter = (ox2 - ox1) * (oy2 - oy1)
        a1 = max(1, (ax2 - ax1) * (ay2 - ay1))
        a2 = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / min(a1, a2)

    def _line_subicon_carried_by_container(self, ix1, iy1, ix2, iy2) -> bool:
        for parent_record in self.foreground_records:
            parent_box = _box4(parent_record)
            if not connector_on_container_border(
                    parent_box, (ix1, iy1, ix2, iy2)):
                continue
            px1, py1, px2, py2 = parent_box
            if foreground_role_for_box(self.icon_probe, px1, py1, px2, py2, self.scale) == "container":
                return True
        return False

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    def _emit_text_entries(self) -> None:
        for i, item in enumerate(self.candidate_texts):
            entry = {
                "id": f"t{i:03d}",
                "type": "text",
                "text": item["text"],
                "bbox": [item["x1"], item["y1"], item["x2"], item["y2"]],
                "confidence": item.get("confidence", 1.0),
            }
            # PaddleOCR per-segment data: chars/char_boxes for pure-CJK,
            # words/word_boxes for every line.
            if "chars" in item and "char_boxes" in item:
                entry["chars"] = list(item["chars"])
                entry["char_boxes"] = [list(b) for b in item["char_boxes"]]
            if "words" in item and "word_boxes" in item:
                entry["words"] = list(item["words"])
                entry["word_boxes"] = [list(b) for b in item["word_boxes"]]
            self.inventory.append(entry)

    def _detect_visual_components(self) -> None:
        # Global contour pass first — surfaces card outlines that a single
        # merged CC would hide.
        for x1, y1, x2, y2, outline_mask in self._detect_outline_rects(
                self.cleaned):
            self._append_outline_record(x1, y1, x2, y2, outline_mask)
        for component_index, (component, label_id) in enumerate(zip(
                self.components, self.component_label_ids)):
            x1, y1, x2, y2, _area = component
            crop = self.cleaned[y1:y2, x1:x2]
            outline_mask = self._detect_outline_mask(
                self.cleaned, x1, y1, x2, y2)
            if outline_mask is not None:
                self._append_outline_record(x1, y1, x2, y2, outline_mask)
                if not (x2 - x1 >= self.img_w * 0.95
                        and y2 - y1 >= self.img_h * 0.95):
                    self._append_foreground_record(x1, y1, x2, y2)
            # Restrict the split projection to this connected component.
            # A shared-mask crop also includes unrelated components that
            # happen to lie inside the component's bounding box; those pixels
            # can fill otherwise-empty projection valleys and suppress splits.
            local_labels = self.component_label_map[y1:y2, x1:x2]
            local_mask = (local_labels == label_id).astype(np.uint8) * 255
            geometry = propose_geometry(
                crop, local_mask, origin=(x1, y1),
                min_area=s_area(300, self.scale),
                ocr_boxes=[(i['x1'], i['y1'], i['x2'], i['y2'])
                           for i in self.ocr_text_items])
            self.geometry_proposal_records.append({
                "parent_bbox": [int(x1), int(y1), int(x2), int(y2)],
                **geometry,
            })
            sub, split_method, split_tree = self._split_component(
                crop, local_mask, origin=(x1, y1))
            self.component_split_records.append({
                "parent_bbox": [int(x1), int(y1), int(x2), int(y2)],
                "split_count": len(sub),
                "split_method": split_method,
                "unsplit_reason": split_tree.get("unsplit_reason"),
                "split_tree": split_tree,
                "split_bboxes": [
                    [int(x1 + sx1), int(y1 + sy1),
                     int(x1 + sx2), int(y1 + sy2)]
                    for sx1, sy1, sx2, sy2 in sub
                ],
            })
            component_role = foreground_role_for_box(
                self.cleaned, x1, y1, x2, y2, self.scale)
            composite_key = None
            is_page_sized = (
                x2 - x1 >= self.img_w * 0.95
                and y2 - y1 >= self.img_h * 0.95
            )
            if (len(sub) > 1 and component_role == "container"
                    and not is_page_sized):
                composite_key = f"component-{component_index:04d}"
                root_method = str(split_tree.get(
                    "nodes", [{}])[0].get("selected_method", ""))
                color_partition = root_method.startswith((
                    "color_seam", "text_rows"))
                self.composite_records.append({
                    "key": composite_key,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "child_bboxes": [
                        [int(x1 + sx1), int(y1 + sy1),
                         int(x1 + sx2), int(y1 + sy2)]
                        for sx1, sy1, sx2, sy2 in sub
                    ],
                    "role": component_role,
                    "color_partition": color_partition,
                })
            if len(sub) == 1:
                cw, ch = x2 - x1, y2 - y1
                if cw >= self.img_w * 0.95 and ch >= self.img_h * 0.95:
                    # Whole-image component → BACKGROUND.
                    self.background_records.append((x1, y1, x2, y2))
                    continue
                self._append_foreground_record(x1, y1, x2, y2)
                self._scan_subicons_inplace(x1, y1, x2, y2)
                self._scan_line_subicons_inplace(x1, y1, x2, y2)
                self._scan_internal_shapes_inplace(x1, y1, x2, y2)
            else:
                for sx1, sy1, sx2, sy2 in sub:
                    ax1, ay1, ax2, ay2 = (
                        x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2)
                    self._append_foreground_record(
                        ax1, ay1, ax2, ay2,
                        force=bool(composite_key and color_partition))
                    if (composite_key and any(
                            _box4(existing) == (ax1, ay1, ax2, ay2)
                            for existing in self.foreground_records)):
                        self.foreground_parent_keys[
                            (ax1, ay1, ax2, ay2)] = composite_key
                    self._scan_subicons_inplace(ax1, ay1, ax2, ay2)
                    self._scan_line_subicons_inplace(ax1, ay1, ax2, ay2)
                    self._scan_internal_shapes_inplace(ax1, ay1, ax2, ay2)

    def _drop_foregrounds_covered_by_shapes(self) -> None:
        """Drop foregrounds whose bbox is largely covered by sub-icons."""
        shape_boxes = (
            [_box4(r) for r in self.subicon_records]
            + [_box4(r) for r in self.line_subicon_records]
            + [_box4(r) for r in self.internal_shape_records]
        )

        def covered(fx1, fy1, fx2, fy2) -> bool:
            farea = max(1, (fx2 - fx1) * (fy2 - fy1))
            for sx1, sy1, sx2, sy2 in shape_boxes:
                ox1, oy1 = max(fx1, sx1), max(fy1, sy1)
                ox2, oy2 = min(fx2, sx2), min(fy2, sy2)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                if (ox2 - ox1) * (oy2 - oy1) >= 0.8 * farea:
                    return True
            return False

        self.foreground_records = [
            r for r in self.foreground_records
            if not covered(*_box4(r))
        ]

    def _drop_outlines_duplicated_by_full_crop(self) -> None:
        """Drop outline masks when an equivalent full crop already exists."""
        def duplicated(ox1, oy1, ox2, oy2) -> bool:
            oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
            for rec in self.background_records + self.foreground_records:
                fx1, fy1, fx2, fy2 = _box4(rec)
                farea = max(1, (fx2 - fx1) * (fy2 - fy1))
                size_ratio = min(oarea, farea) / max(oarea, farea)
                if size_ratio < 0.80:
                    continue
                ix1, iy1 = max(ox1, fx1), max(oy1, fy1)
                ix2, iy2 = min(ox2, fx2), min(oy2, fy2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter >= 0.92 * min(oarea, farea):
                    return True
            return False

        self.outline_records = [
            r for r in self.outline_records
            if not duplicated(r[0], r[1], r[2], r[3])
        ]

    def _coverage_residual_pass(self) -> None:
        """Catch isolated decorations that all detectors missed."""
        text_only = cv2.imread(str(self.text_only_path))
        if text_only is None:
            return
        residual_mask = self._foreground_mask(text_only, self.args.dilate)
        all_bboxes = (
            [_box4(r) for r in self.foreground_records]
            + [_box4(r) for r in self.subicon_records]
            + [_box4(r) for r in self.line_subicon_records]
            + [_box4(r) for r in self.internal_shape_records]
            + [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in self.outline_records]
        )
        covered = np.zeros_like(residual_mask, dtype=np.uint8)
        for bx1, by1, bx2, by2 in all_bboxes:
            covered[by1:by2, bx1:bx2] = 255
        leftover = cv2.bitwise_and(residual_mask, cv2.bitwise_not(covered))
        ln, _, lstats, _ = cv2.connectedComponentsWithStats(leftover, 8)
        residual_min_area = s_area(400, self.scale)
        residual_min_side = s_length(18, self.scale)
        for i in range(1, ln):
            lx, ly, lw, lh, larea = lstats[i]
            if larea < residual_min_area:
                continue
            if lw < residual_min_side and lh < residual_min_side:
                continue
            self._append_foreground_record(
                int(lx), int(ly), int(lx + lw), int(ly + lh))

    def _emit_image_entries(self) -> None:
        """Append visual records to inventory in role-priority order."""
        # Backgrounds first (so they sort to the back via y-position tiebreak).
        for x1, y1, x2, y2 in self.background_records:
            self.inventory.append({
                "id": f"v{self.visual_idx:03d}",
                "type": "image",
                "bbox": [x1, y1, x2, y2],
                "source": "cleaned",
                "role": "background",
            })
            self.visual_idx += 1
        masks_out_dir = (Path(self.args.masks_dir)
                         if self.args.masks_dir else None)
        if self.outline_records and masks_out_dir is None:
            out_path = Path(self.args.out)
            masks_out_dir = out_path.with_name(f"{out_path.stem}_masks")
        if masks_out_dir is not None:
            masks_out_dir.mkdir(parents=True, exist_ok=True)
        # Outlines next so the rounded border renders behind interiors.
        for x1, y1, x2, y2, outline_mask in self.outline_records:
            comp_id = f"v{self.visual_idx:03d}"
            entry = {
                "id": comp_id, "type": "image",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "source": "cleaned", "role": "outline",
            }
            if masks_out_dir is not None:
                full_mask = np.zeros(
                    (self.img_h, self.img_w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = outline_mask
                mask_path = masks_out_dir / f"{comp_id}.mask.png"
                cv2.imwrite(str(mask_path), full_mask)
                entry["mask_path"] = str(mask_path)
            self.inventory.append(entry)
            self.visual_idx += 1
        composite_entries: dict[str, dict] = {}
        composite_children: dict[str, list[str]] = {}
        composite_parent_boxes = {
            tuple(int(v) for v in composite["bbox"]): str(composite["key"])
            for composite in self.composite_records
        }
        partition_child_boxes = {
            tuple(int(v) for v in box)
            for composite in self.composite_records
            if composite.get("color_partition")
            for box in composite["child_bboxes"]
        }
        for composite in self.composite_records:
            comp_id = f"v{self.visual_idx:03d}"
            key = str(composite["key"])
            entry = {
                "id": comp_id,
                "type": "image",
                "bbox": list(composite["bbox"]),
                "source": "cleaned",
                "role": "container",
                "composite_backplate": True,
                "component_key": key,
                "child_bboxes": list(composite["child_bboxes"]),
                "color_partition": bool(composite.get("color_partition")),
            }
            self.inventory.append(entry)
            composite_entries[key] = entry
            composite_children[key] = []
            self.composite_output_ids[key] = comp_id
            self.visual_idx += 1
        for record in self.foreground_records:
            x1, y1, x2, y2 = _box4(record)
            if (x1, y1, x2, y2) in composite_parent_boxes:
                # The separately emitted backplate already represents this
                # root component; do not insert its uncleaned full crop again.
                continue
            comp_id = f"v{self.visual_idx:03d}"
            # Inpainted nested children must crop from the text-only sidecar
            # (strokes intact); other foregrounds crop from cleaned.
            is_nested_child = (x1, y1, x2, y2) in self.inpainted_children
            role_probe = self.source if is_nested_child else self.cleaned
            role = foreground_role_for_box(role_probe, x1, y1, x2, y2, self.scale)
            if (x1, y1, x2, y2) in partition_child_boxes:
                # A rectangular partition tile is a complete visual region,
                # even when its aspect ratio resembles a connector.
                role = "container"
            entry = {
                "id": comp_id, "type": "image",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "source": "source" if is_nested_child else "cleaned",
            }
            if (x1, y1, x2, y2) in partition_child_boxes:
                entry["partition_tile"] = True
            parent_key = self.foreground_parent_keys.get((x1, y1, x2, y2))
            if parent_key and parent_key in self.composite_output_ids:
                entry["parent_id"] = self.composite_output_ids[parent_key]
                entry["component_key"] = parent_key
                composite_children[parent_key].append(comp_id)
            if role:
                entry["role"] = role
            self.inventory.append(entry)
            self.visual_idx += 1
        for key, entry in composite_entries.items():
            entry["child_ids"] = composite_children.get(key, [])
        # Subicons last: they render IN FRONT of their parent card.
        for x1, y1, x2, y2 in self.internal_shape_records:
            self.inventory.append({
                "id": f"v{self.visual_idx:03d}", "type": "image",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "source": "source", "role": "internal",
            })
            self.visual_idx += 1
        for x1, y1, x2, y2, line_mask in self.line_subicon_records:
            comp_id = f"v{self.visual_idx:03d}"
            entry = {
                "id": comp_id, "type": "image",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "source": "source", "role": "line_subicon",
            }
            if masks_out_dir is not None:
                mask_path = masks_out_dir / f"{comp_id}.mask.png"
                cv2.imwrite(str(mask_path), line_mask)
                entry["mask_path"] = str(mask_path)
            self.inventory.append(entry)
            self.visual_idx += 1
        for record in self.subicon_records:
            x1, y1, x2, y2 = _box4(record)
            role = (
                str(record[4])
                if len(record) >= 5 and str(record[4])
                else "subicon"
            )
            self.inventory.append({
                "id": f"v{self.visual_idx:03d}", "type": "image",
                "bbox": [x1, y1, x2, y2],
                "source": "source", "role": role,
            })
            self.visual_idx += 1

    def _emit_geometry_entries(self) -> None:
        """Route supported contours through the same deferred export gate."""
        for record in self.geometry_proposal_records:
            group_boxes = {tuple(b) for g in record.get('groups', [])
                           for b in g['member_bboxes']}
            for candidate in record['candidates']:
                x1, y1, x2, y2 = candidate['bbox']
                evidence = candidate['evidence']
                evidence_families = candidate.get('evidence_families', evidence)
                mask_support = float(candidate.get('component_support_fraction', 0.0))
                supported = (mask_support >= .01 and (
                    len(evidence_families) >= 2 or tuple(candidate['bbox']) in group_boxes
                    or 'hue' in evidence_families
                    or candidate['kind'] == 'ellipse'))
                if not supported:
                    candidate['decision_reason'] = 'single_weak_boundary'
                    candidate['proposal_state'] = 'rejected'
                    continue
                duplicate = False
                for entry in self.inventory:
                    if entry.get('type') != 'image':
                        continue
                    bx1, by1, bx2, by2 = entry['bbox']
                    inter = max(0,min(x2,bx2)-max(x1,bx1))*max(0,min(y2,by2)-max(y1,by1))
                    union = (x2-x1)*(y2-y1)+(bx2-bx1)*(by2-by1)-inter
                    if inter/max(1,union) > .78:
                        duplicate = True
                        break
                if duplicate:
                    candidate['decision_reason'] = 'existing_inventory_object'
                    candidate['proposal_state'] = 'rejected'
                    continue
                points = np.asarray(candidate['contour_points'], np.int32)
                fringe = max(1, s_length(2, self.scale))
                x1, y1 = max(0,x1-fringe), max(0,y1-fringe)
                x2, y2 = min(self.img_w,x2+fringe), min(self.img_h,y2+fringe)
                support = np.zeros((y2-y1,x2-x1),np.uint8)
                contours = candidate.get('support_contours', [candidate['contour_points']])
                cv2.fillPoly(support,[np.asarray(c,np.int32)-np.array([x1,y1])
                                     for c in contours],255)
                support = cv2.dilate(support,np.ones((3,3),np.uint8),iterations=fringe)
                box = (x1,y1,x2,y2)
                self.extraction_masks[box] = (support>0,record['parent_bbox'])
                object_id = f'v{self.visual_idx:03d}'
                self.visual_idx += 1
                candidate_id = (
                    "gc-" + "-".join(str(int(v))
                                     for v in record['parent_bbox']) +
                    f"-{candidate['id']}")
                self.inventory.append({
                    'id': object_id, 'type': 'image', 'bbox': list(box),
                    'source': 'source', 'role': 'geometry',
                    'geometry_kind': candidate['kind'],
                    'geometry_candidate_id': candidate_id,
                    'geometry_parent_bbox': list(record['parent_bbox']),
                    'geometry_evidence': list(candidate.get('evidence', [])),
                    'geometry_evidence_families': list(
                        candidate.get('evidence_families', [])),
                    'component_support_fraction': candidate.get(
                        'component_support_fraction', 0.0),
                })
                candidate['inventory_id'] = object_id
                candidate['geometry_candidate_id'] = candidate_id
                candidate['proposal_state'] = 'queued_for_export_validation'
                candidate['decision_reason'] = 'queued_for_export_validation'

    def _attach_extraction_masks(self) -> None:
        directory = (Path(self.args.masks_dir) if self.args.masks_dir else
                     Path(self.args.out).with_suffix('').with_name(
                         Path(self.args.out).stem + '_masks'))
        directory.mkdir(parents=True, exist_ok=True)
        for entry in self.inventory:
            if entry.get('type') != 'image':
                continue
            box = tuple(entry['bbox'])
            record = self.extraction_masks.get(box)
            if record is None or entry.get('composite_backplate') or entry.get('partition_tile'):
                continue
            support, parent_box = record
            x1, y1, x2, y2 = box
            full = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            full[y1:y2, x1:x2] = support.astype(np.uint8) * 255
            path = directory / f"{entry['id']}.extract.png"
            if not cv2.imwrite(str(path), full):
                raise OSError(f'Cannot write extraction mask: {path}')
            entry['mask_path'] = str(path)
            entry['extraction'] = {
                'state': 'proposed', 'detected_parent_bbox': parent_box,
                'mask_path': str(path), 'exported_object_id': None,
            }
            entry['source'] = 'source'

    def _write_outputs(self) -> None:
        self.inventory.sort(key=_inventory_sort_key)
        out = Path(self.args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.inventory, ensure_ascii=False, indent=2),
            encoding="utf-8")
        # Persist the icon-filled cleaned image so the parent's asset crop
        # comes out without embedded pictograms.
        cv2.imwrite(self.args.clean, self.cleaned)
        text_count = sum(1 for e in self.inventory if e["type"] == "text")
        img_count = sum(1 for e in self.inventory if e["type"] == "image")
        sub_count = sum(1 for e in self.inventory
                        if e.get("role") in {"subicon", "badge_subicon"})
        line_count = sum(1 for e in self.inventory
                         if e.get("role") == "line_subicon")
        print(json.dumps({"text": text_count, "image": img_count,
                          "subicon": sub_count, "line_subicon": line_count,
                          "out": str(out)}, ensure_ascii=False))

    def _write_debug(self) -> None:
        if not self.args.debug_dir:
            return
        debug_dir = Path(self.args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.args.source).stem
        vis = self.source.copy()
        for x1, y1, x2, y2 in self.background_records:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (160, 160, 160), 1)
        for x1, y1, x2, y2, _outline_mask in self.outline_records:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 220), 1)
        for record in self.foreground_records:
            x1, y1, x2, y2 = _box4(record)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 1)
        for it in self.candidate_texts:
            cv2.rectangle(vis, (it["x1"], it["y1"]),
                          (it["x2"], it["y2"]), (0, 200, 0), 1)
        for x1, y1, x2, y2 in [_box4(r) for r in self.subicon_records]:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(vis, "sub", (x1 + 1, max(10, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1,
                        cv2.LINE_AA)
        for x1, y1, x2, y2 in [_box4(r) for r in self.line_subicon_records]:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 0, 180), 2)
            cv2.putText(vis, "line", (x1 + 1, max(10, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 0, 180), 1,
                        cv2.LINE_AA)
        for x1, y1, x2, y2 in self.internal_shape_records:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 100, 255), 2)
            cv2.putText(vis, "int", (x1 + 1, max(10, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1,
                        cv2.LINE_AA)
        legend = [
            ("foreground (img element)", (255, 100, 0)),
            ("background (whole-slide)", (160, 160, 160)),
            ("outline (card border)", (0, 220, 220)),
            ("editable text", (0, 200, 0)),
            ("subicon (movable)", (255, 0, 255)),
            ("line subicon (movable)", (180, 0, 180)),
            ("internal shape (movable)", (0, 100, 255)),
        ]
        for li, (label, col) in enumerate(legend):
            cv2.rectangle(vis, (8, 8 + li * 18), (24, 22 + li * 18), col, -1)
            cv2.putText(vis, label, (28, 20 + li * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                        cv2.LINE_AA)
        cv2.imwrite(str(debug_dir / f"{stem}_inv.png"), vis)

        # _cv2.png — shows the raw foreground mask + every component bbox
        # before filtering. Useful for "missing card border" diagnosis.
        cv2_mask = self.component_mask
        vis2 = self.source.copy()
        mask_red = np.zeros_like(vis2)
        mask_red[..., 2] = cv2_mask
        vis2 = cv2.addWeighted(vis2, 0.4, mask_red, 0.6, 0)
        n_cv, _, stats_cv, _ = cv2.connectedComponentsWithStats(cv2_mask, 8)
        kept = 0
        for i in range(1, n_cv):
            x, y, ww, hh, area = stats_cv[i]
            if area < self.args.min_area:
                continue
            cv2.rectangle(vis2, (int(x), int(y)),
                          (int(x + ww), int(y + hh)), (200, 200, 0), 1)
            kept += 1
        cv2.putText(
            vis2, f"{kept} components (min_area={self.args.min_area})",
            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(debug_dir / f"{stem}_cv2.png"), vis2)

        # Separate the original connected components from the pieces
        # produced by projection splitting. This makes the common failure
        # case (one large merged parent that stayed unsplit) visible without
        # requiring a manual correction UI.
        split_vis = self.source.copy()
        for idx, record in enumerate(self.component_split_records):
            x1, y1, x2, y2 = record["parent_bbox"]
            cv2.rectangle(split_vis, (x1, y1), (x2, y2), (0, 215, 255), 1)
            cv2.putText(split_vis, str(idx), (x1 + 2, max(12, y1 + 13)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (0, 215, 255), 1, cv2.LINE_AA)
            for sx1, sy1, sx2, sy2 in record["split_bboxes"]:
                cv2.rectangle(split_vis, (sx1, sy1), (sx2, sy2),
                              (255, 180, 0), 1)
            tree = record.get("split_tree", {})
            for node in tree.get("nodes", []):
                nx1, ny1, nx2, ny2 = node.get("bbox", [0, 0, 0, 0])
                for candidate in node.get("candidates", []):
                    if candidate.get("accepted"):
                        continue
                    if candidate.get("split_count", 0) <= 1:
                        continue
                    for bx1, by1, bx2, by2 in candidate.get(
                            "split_bboxes", [])[:12]:
                        cv2.rectangle(split_vis, (bx1, by1), (bx2, by2),
                                      (40, 40, 230), 1)
                if node.get("decision_reason") not in {None, "split"}:
                    cv2.putText(
                        split_vis, str(node["decision_reason"]),
                        (nx1 + 2, min(self.img_h - 2, ny1 + 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(debug_dir / f"{stem}_splits.png"), split_vis)

        geometry_vis = self.source.copy()
        for record in self.geometry_proposal_records:
            for candidate in record.get("candidates", []):
                points = np.asarray(candidate.get("contour_points", []),
                                    dtype=np.int32)
                if len(points) >= 3:
                    state = candidate.get("proposal_state")
                    color = ((0, 190, 0) if state == "queued_for_export_validation"
                             else (0, 0, 255) if state == "rejected"
                             else (0, 180, 255))
                    cv2.polylines(geometry_vis, [points.reshape(-1, 1, 2)],
                                  True, color, 1, cv2.LINE_AA)
            for sample in record.get("diagnostics", {}).get(
                    "rejected_samples", []):
                if "bbox" not in sample:
                    continue
                x1, y1, x2, y2 = sample["bbox"]
                cv2.rectangle(geometry_vis, (x1, y1), (x2, y2),
                              (0, 0, 255), 1)
                cv2.putText(geometry_vis, str(sample["reason"])[:18],
                            (x1 + 1, max(10, y1 + 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                            (0, 0, 255), 1, cv2.LINE_AA)
            for group in record.get("groups", []):
                x1, y1, x2, y2 = group["bbox"]
                cv2.rectangle(geometry_vis, (x1, y1), (x2, y2),
                              (0, 180, 255), 2)
                cv2.putText(geometry_vis,
                            f"cards:{group['member_count']}",
                            (x1 + 2, max(12, y1 + 13)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 180, 255), 1, cv2.LINE_AA)
            for connection in record.get("connections", []):
                p1, p2 = connection.get("endpoints", [[], []])
                if len(p1) == 2 and len(p2) == 2:
                    cv2.line(geometry_vis, tuple(p1), tuple(p2),
                             (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(debug_dir / f"{stem}_geometry_proposals.png"),
                    geometry_vis)

        page_area = max(1, self.img_w * self.img_h)
        split_by_parent = {
            tuple(record["parent_bbox"]): record
            for record in self.component_split_records
        }
        geometry_by_parent = {
            tuple(record["parent_bbox"]): record
            for record in self.geometry_proposal_records
        }
        component_report = []
        for x1, y1, x2, y2, area in self.components:
            record = split_by_parent.get((x1, y1, x2, y2), {})
            geometry_record = geometry_by_parent.get((x1, y1, x2, y2), {})
            component_report.append({
                "bbox": [x1, y1, x2, y2],
                "foreground_pixels": int(area),
                "bbox_page_fraction": round(
                    ((x2 - x1) * (y2 - y1)) / page_area, 5),
                "split_count": int(record.get("split_count", 1)),
                "split_method": record.get("split_method", "projection"),
                "unsplit_reason": record.get("unsplit_reason"),
                "role_guess": foreground_role_for_box(
                    self.cleaned, x1, y1, x2, y2, self.scale),
                "split_tree": record.get("split_tree", {}),
                "split_bboxes": record.get("split_bboxes", [[x1, y1, x2, y2]]),
                "geometry_proposals": geometry_record.get("candidates", []),
                "geometry_diagnostics": geometry_record.get("diagnostics", {}),
                "repeated_card_groups": geometry_record.get("groups", []),
                "connections": geometry_record.get("connections", []),
            })
        role_counts: dict[str, int] = {}
        for el in self.inventory:
            if el.get("type") != "image":
                continue
            role = str(el.get("role") or "visual")
            role_counts[role] = role_counts.get(role, 0) + 1
        image_boxes = [
            (int(el["bbox"][0]), int(el["bbox"][1]),
             int(el["bbox"][2]), int(el["bbox"][3]), el.get("id"),
             el.get("role"), el.get("parent_id"),
             el.get("component_key"))
            for el in self.inventory
            if el.get("type") == "image" and len(el.get("bbox", [])) == 4
        ]
        overlap_warnings = []
        for idx, a in enumerate(image_boxes):
            ax1, ay1, ax2, ay2, aid, arole, aparent, _acomponent = a
            aarea = max(1, (ax2 - ax1) * (ay2 - ay1))
            for b in image_boxes[idx + 1:]:
                bx1, by1, bx2, by2, bid, brole, bparent, _bcomponent = b
                barea = max(1, (bx2 - bx1) * (by2 - by1))
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                overlap = (ix2 - ix1) * (iy2 - iy1)
                ratio = overlap / min(aarea, barea)
                if (aparent == bid or bparent == aid
                        or ("outline" in {arole, brole} and ratio >= 0.85)):
                    continue
                if ratio >= 0.60:
                    overlap_warnings.append({
                        "ids": [aid, bid],
                        "roles": [arole, brole],
                        "overlap_bbox": [ix1, iy1, ix2, iy2],
                        "overlap_of_smaller_bbox": round(ratio, 4),
                    })
        large_asset_warnings = [
            {"id": aid, "role": role,
             "bbox": [x1, y1, x2, y2],
             "bbox_page_fraction": round(
                 ((x2 - x1) * (y2 - y1)) / page_area, 5)}
            for x1, y1, x2, y2, aid, role, _parent, _component
            in image_boxes
            if ((x2 - x1) * (y2 - y1)) / page_area >= 0.25
        ]
        proposal_summary = {
            "candidate_count": 0,
            "candidates_by_kind": {},
            "candidates_by_evidence_family": {},
            "candidate_states": {},
            "support_fraction_samples_by_kind": {},
            "decisions_by_reason": {},
            "rejected_by_reason": {},
            "contours_by_evidence": {},
            "contours_truncated_by_evidence": {},
            "repeated_card_group_count": 0,
            "connections_proposed": 0,
            "duplicate_merges": 0,
            "hough_circle_count": 0,
            "candidate_limit_reached_components": 0,
            "components_with_empty_mask": 0,
        }
        for geometry_record in self.geometry_proposal_records:
            diagnostics = geometry_record.get("diagnostics", {})
            proposals = geometry_record.get("candidates", [])
            proposal_summary["candidate_count"] += len(proposals)
            proposal_summary["repeated_card_group_count"] += len(
                geometry_record.get("groups", []))
            proposal_summary["connections_proposed"] += len(
                geometry_record.get("connections", []))
            proposal_summary["duplicate_merges"] += int(
                diagnostics.get("duplicate_merges", 0))
            proposal_summary["hough_circle_count"] += int(
                diagnostics.get("hough_circle_count", 0))
            proposal_summary["candidate_limit_reached_components"] += int(
                bool(diagnostics.get("candidate_limit_reached")))
            proposal_summary["components_with_empty_mask"] += int(
                int(diagnostics.get("component_mask_pixels", 0)) == 0)
            for candidate in proposals:
                kind = str(candidate.get("kind", "unknown"))
                reason = str(candidate.get("decision_reason", "unknown"))
                proposal_summary["candidates_by_kind"][kind] = (
                    proposal_summary["candidates_by_kind"].get(kind, 0) + 1)
                proposal_summary["decisions_by_reason"][reason] = (
                    proposal_summary["decisions_by_reason"].get(reason, 0) + 1)
                state = str(candidate.get("proposal_state", "unknown"))
                proposal_summary["candidate_states"][state] = (
                    proposal_summary["candidate_states"].get(state, 0) + 1)
                for family in candidate.get(
                        "evidence_families", candidate.get("evidence", [])):
                    proposal_summary["candidates_by_evidence_family"][family] = (
                        proposal_summary["candidates_by_evidence_family"].get(
                            family, 0) + 1)
                proposal_summary["support_fraction_samples_by_kind"].setdefault(
                    kind, []).append(float(candidate.get(
                        "component_support_fraction", 0.0)))
            for key in ("rejected_by_reason", "contours_by_evidence",
                        "contours_truncated_by_evidence"):
                for name, value in diagnostics.get(key, {}).items():
                    target = proposal_summary[key]
                    target[name] = target.get(name, 0) + int(value)
        proposal_summary["support_fraction_by_kind"] = {}
        for kind, values in proposal_summary.pop(
                "support_fraction_samples_by_kind").items():
            ordered = sorted(values)
            middle = len(ordered) // 2
            median = (ordered[middle] if len(ordered) % 2 else
                      (ordered[middle - 1] + ordered[middle]) / 2)
            proposal_summary["support_fraction_by_kind"][kind] = {
                "count": len(ordered), "min": round(ordered[0], 4),
                "median": round(median, 4), "max": round(ordered[-1], 4),
            }
        report = {
            "schema_version": 2,
            "source": str(Path(self.args.source).resolve()),
            "image_size": [self.img_w, self.img_h],
            "method": "OpenCV canvas/Lab/local-contrast/Canny + component-specific masks + recursive multi-scale morphology opening + projection valleys + component-owned multi-channel geometry proposals",
            "geometry_proposals": {
                "schema_version": 2,
                **proposal_summary,
                "export_enabled": True,
                "ownership_evidence": "candidate contour/region support is measured against its connected-component mask",
                "ownership_rules": {
                    "min_region_support_fraction": 0.01,
                    "min_boundary_support_fraction": 0.04,
                    "boundary_band_width_px": 3,
                    "min_boundary_support_pixels": "max(3, round(contour_perimeter * 0.02))",
                },
                "note": "Proposal rejections and contour truncation are counted; supported contours enter deferred mask export, and layout extraction_audit records acceptance or retention in the parent.",
            },
            "counts": {
                "connected_components": len(self.components),
                "composite_backplates": len(self.composite_records),
                "linked_composite_children": sum(
                    1 for key in self.foreground_parent_keys.values()
                    if key in self.composite_output_ids),
                "inventory_images": sum(
                    1 for el in self.inventory if el.get("type") == "image"),
                "inventory_text": sum(
                    1 for el in self.inventory if el.get("type") == "text"),
                "images_by_role": role_counts,
            },
            "large_unsplit_components": [
                component for component in component_report
                if component["bbox_page_fraction"] >= 0.08
                and component["split_count"] == 1
            ],
            "large_asset_warnings": large_asset_warnings,
            "overlap_warnings": overlap_warnings,
            "components": component_report,
        }
        (debug_dir / f"{stem}_segmentation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8")
        cv2.imwrite(str(debug_dir / f"{stem}_foreground_mask.png"), cv2_mask)

    def build_and_write(self) -> None:
        self._emit_text_entries()
        self._detect_visual_components()
        self._drop_outlines_duplicated_by_full_crop()
        self._coverage_residual_pass()
        self._emit_image_entries()
        self._emit_geometry_entries()
        self._attach_extraction_masks()
        self._write_outputs()
        self._write_debug()

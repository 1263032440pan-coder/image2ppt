## v0.5 · Recover sharpness without SLIC grid artefacts

- Remove 640px colour downsample + bilateral + Gaussian blur + upscale.
- Use source-resolution measurements with one-pixel scanlines and adaptive gradient stops at source colour edges.
- Extract Detail IR at 1000px instead of 800px; preserve narrow ink with less polygon simplification.
- Default to 20% extra-stroke overlay to prevent doubled/dark/dirty contours (configurable DETAIL_OVERLAY_STRENGTH=0..1).
- Add sharp 1px line and non-grid regression tests.
- Source-guided SVG gradients remain a trace-based approximation, not VLM semantic procedural illustration.

## v0.4 · Remove artificial color-grid artefacts

- Replaced rendered SLIC superpixel polygons (and same-color `stroke-width=1.3`) with a fully vector, spatially measured continuous SVG gradient field.
- SVG Color IR is encoded as narrow horizontal gradients with adaptive stops near strong real image color transitions; no raster `<image>` and no colored cell outlines.
- Local color refinements now increase gradient sampling density; they never layer extra color polygons.
- Split long spurious contour chords produced by edge-path decimation, reducing unwanted diagonals.
- Added anti-grid and color-position tests; preserved geometry, agent orchestration, and downloadable SVG/PNG outputs.
- Known limitation: gradient strips represent image-derived color samples, not a generative semantic painting program.

# Changelog

## v0.3 · Spatial Color / Detail reconstruction

- New `app/color.py`: full-scene SLIC superpixel Color IR, measured RGB fills, native editable polygon paths, spatial owners.
- New `app/detail.py`: source edge-to-polyline Detail IR for all regions, not just koi.
- Expanded `app/geometry.py`: more candidate object kinds and optional Hough-based structure parts.
- New `app/segmentation.py`: conservative estimated visible masks with foreground occlusion handling.
- New `app/region_metrics.py`: aligned CIEDE2000, edge F1, heatmap and explicit bbox-vs-mask labeling.
- New `app/repair.py`: local color-field subdivision for the lowest-scoring band.
- New subprocess workers for whole jobs and SVG rasterization; 1 active job limit for local desktop use.
- Web UI adds Color/Detail IR switches and downloadable region reports.
- No source bitmap is embedded in SVG. Full-scene SLIC trace improves color placement but **does not** amount to fully semantic graphics-code reconstruction.
- Real third-party multimodal API not tested; demo mode uses a fixed scene.

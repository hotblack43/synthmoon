# Changelog

## 0.3.1 - 2026-02-28
- Started item 5 work: added DEM-aware extended-Sun visibility correction for `shadows.sun = "dem"` (first pass).
- Added `sun.shadow_disk_samples` support for extended-Sun DEM shadow sampling control.

## 0.3.0 - 2026-02-28
- Added side-by-side comparison mode that renders `advanced` and `legacy_parallel` outputs in one run.
- Added diff cube products with absolute and relative comparison layers.
- Added robust percent-difference layers for easier interpretation in low-signal regions.
- Added true point-source Earthshine legacy mode (`EARTHPT=1`) so comparisons target extended-Earth vs point-Earth physics.
- Added run metadata fields in FITS headers to track comparison mode choices.

## 0.2.0
- Baseline v0 renderer with SPICE geometry, Sun + Earthlight contributions, and FITS cube output.

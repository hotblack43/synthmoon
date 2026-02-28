# synthmoon v0

This is a **version 0** reference implementation for generating a synthetic 512×512 Moon image (1° FOV)
with **Lambert** reflectance, **finite-distance SPICE geometry**, and optional **extended-source earthlight**.

It writes a 16-bit FITS primary image using BSCALE/BZERO and (optionally) a float32 extension.
Version notes are tracked in `CHANGELOG.md`.

## Quick start

### 1) Create an environment (uv or pip)

Using uv (recommended):
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Using pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Download SPICE generic kernels

```bash
python scripts/download_kernels.py
```

This downloads:
- `naif0012.tls` (leapseconds) from NAIF generic kernels. 
- `pck00011.tpc` (text PCK: body constants and IAU frames).
- `de442s.bsp` (planetary ephemerides SPK; includes Earth/Moon/Sun).

Sources (NAIF):
- LSK directory listing includes `naif0012.tls`. 
- PCK `pck00011.tpc` is in NAIF generic kernels.
- SPK planets directory listing includes `de442s.bsp`.

### 3) Edit the config

Edit `scene.toml`.

### 4) Render

```bash
synthmoon-render --config scene.toml
# or:
python -m synthmoon.run_v0 --config scene.toml
```

Output FITS is written to `OUTPUT/` by default.

Regression check for earthlight layers (IF_EARTH/RAD_EAR non-zero at a known UTC):
```bash
./tools/regression_check_earthlight_nonzero.sh
```

Moon BRDF switch:
```toml
[moon]
brdf = "lambert"   # or "hapke"

[moon.hapke]
single_scattering_albedo = 0.55
phase_b = 0.30
phase_c = 0.40
opposition_b0 = 1.00
opposition_h = 0.06
roughness_deg = 20.0
```
`hapke` modifies direct-sun lunar reflectance (I/F) while keeping output layer layout unchanged.

Lambert vs Hapke regression check:
```bash
./tools/regression_compare_lambert_hapke.sh
```

Advanced vs legacy-parallel comparison (two outputs + optional diff in one run):
```toml
[comparison]
enabled = true
write_diff = true
pct_floor_if = 1.0e-5   # percent diff only where |IFTOTAL_advanced| > floor
pct_clip_percent = 1.0  # clipped display-friendly percent layer (+/- this value)
advanced_suffix = "_advanced"
legacy_suffix = "_legacy_parallel"
diff_suffix = "_diff_if"
```
With `enabled = true`, one render command writes:
- `<out>_advanced.fits` (your current model)
- `<out>_legacy_parallel.fits` (point-source Sun + point-source Earth)
- `<out>_diff_if.fits` (IFTOTAL difference: advanced - legacy, if `write_diff=true`)

High-resolution render then downsample (recommended for cleaner 512x512 products):
```toml
[camera]
nx = 2048
ny = 2048

[output]
downsample_to_nx = 512
downsample_to_ny = 512
```
Downsampling is NaN-aware block averaging and is applied to all physical output layers.

Direct MP4/MOV build (no giant movie FITS), e.g. 60 days at 6-hour steps:
```bash
source .venv/bin/activate
python tools/build_movie_video_hourly.py \
  --config /tmp/scene_movie_hires.toml \
  --start-utc 2006-02-13T06:18:45Z \
  --hours 240 \
  --step-hours 6 \
  --fps 24 \
  --out-mp4 OUTPUT/movie_60d_6h_iftotal.mp4 \
  --out-mov OUTPUT/movie_60d_6h_iftotal.mov \
  --workdir /tmp/synthmoon_movie_frames \
  --tmp-fits /tmp/synthmoon_movie_frame.fits \
  --crf 18
```

## Notes

- Geometry is computed using SPICE via SpiceyPy. Best practice is to load kernels via a meta-kernel (FURNSH).  
- Earthlight is treated as an **extended source** (Earth disk quadrature) and is tile-cached for speed.
- This is v0: Moon DEM/albedo maps are stubbed as constant fields; hooks exist to upgrade to real maps.

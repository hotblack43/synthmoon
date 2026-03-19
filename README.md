# synthmoon v0

This is a **version 0** reference implementation for generating synthetic Moon images (including 512x512 products)
with **finite-distance SPICE geometry**, Moon BRDF options (**Lambert** or **Hapke**), optional **LOLA DEM**
terrain/shadowing, and optional **extended-source earthlight**.

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
uv run python scripts/download_kernels.py
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

Camera pointing modes:
```toml
[camera]
pointing = "moon_center"      # default
# or: "moon_limb_east" | "moon_limb_west" | "moon_limb_north" | "moon_limb_south" | "moon_limb_prograde" | "orbit_path"
up_mode = "north"             # or "surface_down" for outreach framing
limb_offset_scale = 1.0       # 1.0 = limb
```

### 4) Render

```bash
synthmoon-render --config scene.toml
# or:
uv run python -m synthmoon.run_v0 --config scene.toml
```

Output FITS is written to `OUTPUT/` by default.

Standard Moon-image renders now apply a default Gaussian PSF before the final output downsampling.
The default is `psf_mode = "gaussian"` with `psf_fwhm_out_px = 0.85` in `scene.toml`.
Written Moon FITS headers record `PSFMODE`, `PSFSIG`, `PSFFWHM`, `PSFOUT`, and `SUMPIX`.

### 5) Single image from Earth lon/lat at a given Julian Day

The renderer accepts UTC as input (`time.utc` or `--utc`).  
Use Earth-site observer coordinates in `scene.toml`:

```toml
[observer]
mode = "earth_site"
lon_deg = -155.5763
lat_deg = 19.5362
height_m = 3397.0
```

Run one image for a specific UTC:

```bash
uv run python -m synthmoon.run_v0 \
  --config scene.toml \
  --utc 2006-02-17T06:18:45Z \
  --out OUTPUT/synth_moon_single.fits
```

If your input is a Julian Day, convert JD -> UTC first, then render:

```bash
JD=2453783.7623264
UTC=$(uv run python -c "from astropy.time import Time; print(Time(${JD}, format='jd', scale='utc').isot + 'Z')")
uv run python -m synthmoon.run_v0 --config scene.toml --utc "$UTC" --out OUTPUT/synth_moon_from_jd.fits
```

`run_v0` writes `DATE-OBS` and `JD-OBS` in FITS headers, so the exact UTC/JD used is recorded in output.

Advanced Moon render (Hapke + DEM + DEM solar shadows) from Earth lon/lat:

```toml
[moon]
brdf = "hapke"
dem_fits = "DATA/moon_dem_lola_ldem16_m.fits"
dem_refine_iter = 3

[sun]
extended_disk = true

[shadows]
mode = "dem"
sun = "dem"
```

```bash
uv run python -m synthmoon.run_v0 \
  --config scene.toml \
  --utc 2006-02-17T06:18:45Z \
  --out OUTPUT/synth_moon_advanced_dem_hapke.fits
```

One-command wrapper (UTC or JD + lon/lat/alt from CLI):

```bash
tools/go_single_image.sh \
  --lon -155.5763 --lat 19.5362 --alt-m 3397 \
  --jd 2453783.7623264 \
  --out OUTPUT/synth_moon_single_from_cli.fits
```

One-command Moon/Earth pair wrapper (EO-aware):

```bash
tools/go_earth_moon_pair.sh \
  --lon -155.5763 --lat 19.5362 --alt-m 3397 \
  --jd 2455748.7651276 \
  --out-dir OUTPUT
```

This script:
- renders the matching Earth and Moon pair for the given UTC/JD and observer site
- uses the recommended EO Earth workflow
- checks whether the needed daily/yearly EO products are already on disk
- if any EO input is missing, it stops and prints the exact `uv run` commands needed to download/extract the missing files

Synthetic Earth FITS (for Earthlight diagnostics):

```bash
uv run python tools/render_earth_fits.py \
  --config scene.toml \
  --utc 2006-02-17T06:18:45Z \
  --nx 1024 --ny 1024 \
  --out OUTPUT/earth_synth_20060217T061845Z.fits
```

This writes a float64 FITS cube with scalar and RGB layers, including:
`RAD_EAR, IF_EARTH, RAD_R, RAD_G, RAD_B, IF_R, IF_G, IF_B, A_EFF, AEFF_R, AEFF_G, AEFF_B, A_SURF, ASRF_R, ASRF_G, ASRF_B, CLOUDF, MU0, MUV, FSUN, ELON, ELAT, MASK`
and `ECLASS` when a class map is active.
(`RAD_EAR` is layer 1, and `RAD_R/G/B` carry the color-capable Earth radiance channels.)

When the first-order Earth atmosphere is enabled, the cube also includes atmosphere diagnostics such as:
`RAD_SURF, IF_SURF, RAD_ATM, ATM_R, ATM_G, ATM_B, ATMTOT, ATMT_R, ATMT_G, ATMT_B`.
In that mode, `RAD_EAR` / `RAD_R/G/B` are atmosphere-inclusive totals.

Earth glint model switch in `scene.toml`:

```toml
[earth]
ocean_glint_model = "cox_munk"  # "simple" | "cox_munk"
ocean_wind_m_s = 6.0
ocean_refractive_index = 1.334
ocean_glint_strength = 0.15
```

First-order Earth atmosphere switch in `scene.toml`:

```toml
[earth]
atmosphere_enable = true
atmosphere_strength = 0.12
rayleigh_tau_rgb = [0.10, 0.06, 0.03]
aerosol_tau_rgb = [0.02, 0.02, 0.018]
aerosol_g = 0.70
```

This is a first-pass physically motivated model:
- surface radiance is attenuated by atmospheric transmission
- single-scattered Rayleigh + aerosol path radiance is added in RGB
- the effect is strongest toward the limb and near the illuminated edge, not as a generic screen-space haze

Optional Earth surface-class workflow (0=ocean, 1=land, 2=ice):

```bash
uv run python tools/make_earth_class_map_from_albedo.py \
  --in-fits DATA/earth_albedo.fits \
  --out-fits DATA/earth_class_map.fits
```

Then enable in `scene.toml`:

```toml
[earth]
class_map_fits = "DATA/earth_class_map.fits"
class_map_lon_mode = "-180_180"
class_map_interp = "nearest"
class_ocean_values = [0]
class_land_values = [1]
class_ice_values = [2]
```

Important:
- this helper currently builds a simple class map from the existing albedo map
- it is useful for testing class-driven logic
- it is not the recommended default for the EO workflow described below
- for the EO workflow, leave `class_map_fits = ""` unless you have built a proper EO-derived class map

EO-derived class map from MODIS land cover:

```bash
uv run python tools/make_earth_class_map_from_modis_landcover.py \
  --in-hdf 'DATA/MODIS/MCD12C1*.hdf' \
  --scheme modis_igbp \
  --out-fits DATA/MODIS/earth_landcover_modis_2011.fits
```

This preserves the raw MODIS LC_Type1 / IGBP class ids so the renderer can distinguish
water, forest, shrubland, grassland, cropland, urban, permanent snow/ice, barren desert, etc.
The default `modis_igbp` RGB preset then assigns approximate visible reflectance colors per class.

Then enable it explicitly:

```toml
[earth]
class_map_fits = "DATA/MODIS/earth_landcover_modis_2011.fits"
class_map_lon_mode = "0_360"
class_map_interp = "nearest"
class_ocean_values = [0]
class_land_values = []
class_ice_values = [15]
class_rgb_preset = "modis_igbp"
```

Use this if you want richer surface typing and color-capable Earth products. The simpler EO default still works without any class map.

Earthdata token setup for EO downloads:

```bash
export EARTHDATA_TOKEN='your_very_long_token_here'
echo ${#EARTHDATA_TOKEN}
```

Add that `export` line to `~/.bashrc` if you want it available in future shells.

MODIS daily global cloud maps (recommended over single `MOD06_L2` swaths):

```bash
uv run python tools/download_modis_cloud_granule.py \
  --utc 2011-07-06T06:21:47Z \
  --product MOD08_D3 \
  --out-dir DATA/MODIS
```

Then extract cloud maps from the downloaded daily HDF:

```bash
uv run python tools/extract_modis_l3_cloud_maps.py \
  --in-hdf 'DATA/MODIS/MOD08_D3*.hdf' \
  --out-dir DATA/MODIS
```

This is a separate two-step workflow:
- step 1 downloads the MODIS daily HDF from Earthdata
- step 2 extracts lon/lat FITS maps from that HDF

Point `scene.toml` at the extracted daily cloud maps:

```toml
[earth]
cloud_fraction_map_fits = "DATA/MODIS/<daily_prefix>_cloud_fraction.fits"
cloud_tau_map_fits = "DATA/MODIS/<daily_prefix>_cloud_tau.fits"
cloud_map_lon_mode = "-180_180"
cloud_map_interp = "nearest"
```

This daily Level-3 path avoids the narrow-strip problem of a single `MOD06_L2` swath.

NSIDC daily sea-ice maps (sea ice only; not Greenland/Antarctic land ice):

```bash
uv run python tools/download_nsidc_g02202_daily.py \
  --utc 2011-07-06T06:21:47Z \
  --hemisphere both \
  --out-dir DATA/NSIDC
```

Then convert the two polar daily files into one global lon/lat FITS ice-fraction map:

```bash
uv run python tools/extract_nsidc_g02202_ice_map.py \
  --north-nc DATA/NSIDC/sic_psn25_20110706_F17_v05r00.nc \
  --south-nc DATA/NSIDC/sic_pss25_20110706_F17_v05r00.nc \
  --out-fits DATA/NSIDC/ice_fraction_20110706.fits
```

This is also a separate two-step workflow:
- step 1 downloads the north and south NSIDC daily files
- step 2 reprojects them into one equirectangular FITS map

Enable that sea-ice map in `scene.toml`:

```toml
[earth]
ice_fraction_map_fits = "DATA/NSIDC/ice_fraction_20110706.fits"
ice_map_lon_mode = "-180_180"
ice_map_interp = "nearest"
ice_fraction_threshold = 0.15
ice_fraction_blend = true

# keep fake caps off when using real sea ice:
class_ice_values = []
seasonal_ice_enable = false
```

Important:
- `MOD08_D3` gives daily global cloud fields
- `G02202_V5` gives daily sea-ice concentration over ocean
- neither product gives permanent land ice on Greenland or Antarctica

Static land ice for Greenland and Antarctica:

Use an annual MODIS land-cover product with a permanent snow/ice class. The easy global first choice is `MCD12C1`.

Download the annual land-cover file:

```bash
uv run python tools/download_modis_landcover_file.py \
  --year 2011 \
  --product MCD12C1 \
  --out-dir DATA/MODIS
```

Then extract a global land-ice mask FITS from the MODIS land-cover HDF:

```bash
uv run python tools/extract_modis_landice_mask.py \
  --in-hdf 'DATA/MODIS/MCD12C1*.hdf' \
  --out-fits DATA/MODIS/land_ice_mask_2011.fits
```

This is again a separate two-step workflow:
- step 1 downloads the annual MODIS land-cover HDF
- step 2 extracts a global 0..1 FITS mask for permanent land ice

Enable that static land-ice mask in `scene.toml`:

```toml
[earth]
land_ice_mask_fits = "DATA/MODIS/land_ice_mask_2011.fits"
land_ice_mask_lon_mode = "-180_180"
land_ice_mask_interp = "nearest"
land_ice_mask_threshold = 0.5
land_ice_mask_blend = true

# keep fake caps off when using real products:
class_ice_values = []
seasonal_ice_enable = false
```

The extractor assumes the standard IGBP permanent snow/ice class value (`15`) from the MODIS LC_Type1 classification. You can override that on the command line if needed.

A practical split is:
- use `MCD12C1` for static land ice
- use `G02202_V5` for daily sea ice
- use `MOD08_D3` for daily clouds

Current recommended EO-Earth default:

```toml
[earth]
albedo_map_fits = "DATA/earth_albedo.fits"
class_map_fits = ""

cloud_fraction_map_fits = "DATA/MODIS/<daily_prefix>_cloud_fraction.fits"
cloud_tau_map_fits = "DATA/MODIS/<daily_prefix>_cloud_tau.fits"
cloud_map_lon_mode = "-180_180"

ice_fraction_map_fits = "DATA/NSIDC/ice_fraction_YYYYMMDD.fits"
ice_map_lon_mode = "-180_180"
ice_fraction_blend = true

land_ice_mask_fits = "DATA/MODIS/land_ice_mask_YYYY.fits"
land_ice_mask_lon_mode = "-180_180"
land_ice_mask_blend = true

seasonal_ice_enable = false
class_ice_values = []
```

This keeps Earth land-surface classification possible, but separate from the old toy class map:
- if you want richer land-cover-aware color, provide a proper EO-derived class map and set `class_map_fits`
- if you do not have that yet, the recommended path is albedo + clouds + daily sea ice + static land ice, with `class_map_fits = ""`

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

Earth-only color movie over a JD range, with optional black padding around the disk:
```bash
uv run python tools/build_earth_color_movie_jd.py \
  --config scene.toml \
  --start-jd 2453789.7630208 \
  --end-jd 2453790.7630208 \
  --step-hours 0.3333333333 \
  --nx 1024 --ny 1024 \
  --pad-frac 0.10 \
  --out-mp4 OUTPUT/earth_movie_jd_color.mp4
```

This uses the Earth RGB radiance layers (`RAD_R/G/B`) from `tools/render_earth_fits.py`
and writes a color MP4. If needed, flip the result for playback with:

```bash
ffmpeg -y -i OUTPUT/earth_movie_jd_color.mp4 -vf vflip \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  OUTPUT/earth_movie_jd_color_vflip.mp4
```

## Notes

- Geometry is computed using SPICE via SpiceyPy. Best practice is to load kernels via a meta-kernel (FURNSH).  
- Earthlight is treated as an **extended source** (Earth disk quadrature) and is tile-cached for speed.
- This is v0: Moon DEM/albedo maps are stubbed as constant fields; hooks exist to upgrade to real maps.

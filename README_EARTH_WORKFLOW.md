# Earth Workflow

This note records the current recommended Earth-rendering and Earth-movie path in `synthmoon`.

## Current Earth Render Path

Use:

```bash
uv run python tools/render_earth_fits.py \
  --config scene.toml \
  --jd 2453789.7630208 \
  --nx 1024 --ny 1024 \
  --out OUTPUT/earth_synth_example.fits
```

This writes a float64 FITS cube containing:

- scalar layers such as `RAD_EAR`, `IF_EARTH`, `A_EFF`, `A_SURF`, `CLOUDF`, `MU0`, `MUV`, `FSUN`, `ELON`, `ELAT`, `MASK`
- RGB Earth radiance and reflectance layers:
  - `RAD_R`, `RAD_G`, `RAD_B`
  - `IF_R`, `IF_G`, `IF_B`
  - `AEFF_R`, `AEFF_G`, `AEFF_B`
  - `ASRF_R`, `ASRF_G`, `ASRF_B`
- `ECLASS` when a class map is active

If the first-order atmosphere is enabled, the cube also contains:

- `RAD_SURF`, `IF_SURF`
- `RAD_ATM`
- `ATM_R`, `ATM_G`, `ATM_B`
- `ATMTOT`, `ATMT_R`, `ATMT_G`, `ATMT_B`

`RAD_R/G/B` are the current color-capable Earth radiance layers and are the preferred basis for Earth quicklooks and Earth movies.

## Current Earth Inputs

The current recommended config path uses:

- `DATA/earth_albedo.fits`
- MODIS daily cloud fraction / cloud tau maps
- NSIDC daily sea-ice fraction map
- MODIS static land-ice mask
- MODIS land-cover class map, preserved as raw IGBP class ids

The class-driven color preset is:

```toml
[earth]
class_rgb_preset = "modis_igbp"
```

This is intended to make desert / vegetation / water / snow-ice distinctions visible in Earth images and later in colored Earthshine work.

## First-Order Atmosphere

The current Earth renderer now has a first-pass atmosphere model:

- atmospheric transmission on the Sun and view paths
- single-scattered Rayleigh path radiance
- single-scattered aerosol path radiance using a Henyey-Greenstein phase function
- RGB output so the limb can go bluish in a physically motivated way

Current default knobs in `scene.toml` are:

```toml
[earth]
atmosphere_enable = true
atmosphere_strength = 0.12
atmosphere_mu_floor = 0.03
atmosphere_limb_boost_power = 0.15
atmosphere_twilight_mu = 0.12
atmosphere_sky_rgb = [0.60, 0.72, 1.00]
rayleigh_tau_rgb = [0.10, 0.06, 0.03]
aerosol_tau_rgb = [0.02, 0.02, 0.018]
aerosol_g = 0.70
```

This is still a first-order approximation, not a full multiple-scattering atmosphere.

## Preferred Earth Movie Path

For a static-config movie over a JD range, use:

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

This:

- renders Earth-only frames over a JD range
- uses `RAD_R/G/B`
- converts them to RGB PNG frames
- adds black padding if requested
- encodes an MP4 with `ffmpeg`

For the EO-aware fast movie path with daily cloud and ice updates, use:

```bash
uv run python tools/build_earth_color_movie_jd_eo_fast.py \
  --config scene.toml \
  --start-jd 2453789.7630208 \
  --end-jd 2453791.7630208 \
  --step-hours 0.1666666667 \
  --nx 1024 --ny 1024 \
  --pad-frac 0.10 \
  --out-mp4 OUTPUT/earth_movie_2d_10min_eo_fast_atm_on.mp4 \
  --fetch-missing
```

This EO-aware path:

- rewrites the daily MODIS cloud fraction / tau inputs
- rewrites the daily NSIDC sea-ice input
- rewrites the yearly land-ice mask input
- preserves `class_map_fits`, so the MODIS IGBP land-cover colour model remains active
- forces `seasonal_ice_enable = false` so synthetic seasonal caps do not double-count real sea ice
- keeps rendering in one Python process and avoids per-frame FITS write/read
- if `--keep-frames` is omitted, removes its temporary PNG/config workdir after the MP4 is written

By default it also writes a compact table with:

- `jd`
- `sum_r`
- `sum_g`
- `sum_b`

and then generates:

- `..._ratio_combined.png`
- `..._mag_combined.png`

If the movie orientation needs flipping for playback:

```bash
ffmpeg -y -i OUTPUT/earth_movie_jd_color.mp4 -vf vflip \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  OUTPUT/earth_movie_jd_color_vflip.mp4
```

## Playback

A working command from the repo root is:

```bash
mplayer -vo gl -loop 0 OUTPUT/earth_movie_jd_color_vflip.mp4
```

## Important Caveat

There is a known longitude-convention wrinkle in the MODIS-derived static land-ice products used during this work.

Practical rule:

- trust the current renderer and current committed config/code together
- if Antarctica or land-cover placement looks wrong after future map regeneration, re-check the effective longitude convention of:
  - `earth_landcover_modis_2011.fits`
  - `land_ice_mask_2011.fits`

The current committed renderer contains defensive handling so Antarctic land ice does not fall through to dark ocean because of that mismatch.

## Recommended “Known Good” Outputs From This Session

- `OUTPUT/earth_synth_jd2453789p7630208_color.fits`
- `OUTPUT/earth_synth_jd2453789p7630208_color.png`
- `OUTPUT/earth_movie_jd2453789_24h_20min_color_pad_vflip.mp4`

## Plotting From Movie-Time RGB CSV

Use:

```bash
uv run python tools/plot_earth_rgb_simple_csv.py \
  --csv OUTPUT/earth_movie_2d_10min_eo_fast_atm_on_rgb.csv \
  --out-prefix OUTPUT/earth_movie_2d_10min_eo_fast_atm_on
```

This writes:

- `OUTPUT/earth_movie_2d_10min_eo_atm_on_ratio_combined.png`
- `OUTPUT/earth_movie_2d_10min_eo_atm_on_mag_combined.png`

The magnitude-style plot uses:

- `$m_R - m_G$`
- `$m_R - m_B$`
- `$m_B - m_G$`

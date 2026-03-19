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

## Preferred Earth Movie Path

Use the dedicated Earth color movie builder:

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


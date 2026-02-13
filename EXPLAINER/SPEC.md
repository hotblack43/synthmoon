# SPEC.md — SYNTHMOON: Synthetic Moon Images with Sunlight + Earthlight (Earthshine)

## Goal
Build a modular Python renderer that generates synthetic **512×512** Moon images with **1°×1°** field of view, as seen from a specified observer (Earth site or spacecraft). Illumination includes **direct sunlight** and **earthlight** (sunlight reflected by Earth). Geometry must come from **SPICE**.

## Non-negotiables
- One UTC timestamp per image; store **full Julian Date** in FITS header as **JD-OBS** formatted **f15.7** (no MJD).
- Compute in **floating point** throughout (earthshine is faint).
- Output is **DS9-friendly FITS** and must not silently quantise to int16.

## Current v0.1.x behaviour (what exists now)
### Geometry
- Uses SpiceyPy with a small generic NAIF kernel set (LSK + PCK + planetary SPK).
- Observer:
  - Earth site lon/lat/height OR spacecraft inertial state.
- Camera:
  - Pinhole/gnomonic rays; pointing set to Moon centre; roll supported.

### Moon surface
- Base surface is a sphere of radius `moon.radius_km`.
- **Lunar albedo map** supported via an equirectangular FITS (e.g., LROC WAC 643 nm mosaic converted to FITS).
- **Orography (LOLA DEM)**:
  - LOLA LDEM_16 global DEM converted to FITS.
  - Ray–Moon hit is refined using a “sphere + DEM displacement” iteration.
  - Surface normals derived from DEM slopes (finite differences) for Lambert shading.
  - Diagnostic layers: elevation (metres) and slope (degrees).

### Illumination
- Sunlight and earthlight can be toggled independently.
- **Earthlight** treats Earth as an extended disk and integrates over disk samples; uses tile caching for speed.
- **Sun as extended disk** (finite solar diameter):
  - Adaptive disk sampling only in the narrow partial-visibility band near terminator/horizon.
  - Else point-source Sun is used for speed.

### Shadowing
- v0.1.x uses simple geometric horizon tests.
- Planned: DEM-based terrain self-shadowing for both Sun and Earthlight directions (hard shadows first, then penumbra).

### Output format
- Writes a single **FITS cube in the PRIMARY HDU** (NAXIS=3). Layer names are stored as `LAY1..LAYN` keywords.
- Primary layers are stored as **float32/float64**; an optional “scaled for viewing” layer maps values to **0..65535** (still float).
- Typical layers include:
  - SCALED (0..65535 float for viewing)
  - IFTOTAL, IF_SUN, IF_EARTH (dimensionless I/F)
  - RADTOT, RAD_SUN, RAD_EAR (broadband radiance, W m⁻² sr⁻¹; TSI-based)
  - ALBMOON (albedo actually used per pixel)
  - ELEV_M (metres) and SLOPDEG (degrees) when DEM is enabled
- FITS headers include JD-OBS, DATE-OBS, kernel provenance, and key config settings.

## Performance principles
- Early-out for rays that miss the Moon (bounding sphere).
- Optional tile caching for earthlight disk integration.
- Adaptive sampling for extended Sun (only near terminator).

## Planned upgrades
1) Terrain self-shadowing (DEM horizon/ray test) for Sun and Earthlight.
2) Hapke photometry for the Moon (spatially varying parameter maps later).
3) Spectral/filtered rendering: wavelength grid + solar spectrum + wavelength-dependent BRDFs.
4) Better Earth model: map-based albedo + clouds + later glint and spectral reflectance.

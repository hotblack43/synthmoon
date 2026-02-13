# SPEC.md — Synthetic Moon Images with Sunlight + Earthlight (Earthshine)

## Goal
Create a modular Python renderer that generates synthetic 512×512 monochrome (initially) images of the Moon as seen from a specified observer in space or on Earth. Illumination must include **direct sunlight** and **earthlight** (sunlight reflected by Earth). Geometry must come from **celestial mechanics** (SPICE). No parallel-ray approximations: all illumination directions derived from finite-distance bodies.

## Key Requirements
- **Image geometry**
  - Image size: **512 × 512**
  - Field of view: **1.0° × 1.0°** (square)
  - Camera model: pinhole/gnomonic projection (upgradeable)
- **Time handling**
  - One UTC timestamp per image (exposures are milliseconds)
  - Store **full Julian Date** in FITS header as **f15.7**: `JD-OBS`
  - Do **not** use MJD anywhere
- **Numeric pipeline**
  - Compute radiance/intensity in **floating point** throughout (float32/float64)
  - Earthshine is faint: avoid 8-bit rendering
- **Output**
  - Store outputs as **16-bit FITS** images
  - Use **BSCALE/BZERO** to preserve floating-point physical values when storing as int16
  - Include an **extensive FITS header** with full provenance and settings
- **Physics / realism**
  - Moon surface includes **lunar orography** (DEM-based)
  - Moon reflectance: **Lambert first**, later upgradeable to **Hapke**
  - Earth reflectance: **map-based albedo** + simple reflectance (Lambert first), later spectral/BRDF upgrades
  - Earth is an **extended source**: Earthlight computed as a disk integral; visible Earth varies across lunar surface points
  - Shadowing:
    - v0: simple (sign tests / optional coarse terrain checks)
    - later: upgradeable to better terrain self-shadowing (horizon maps / ray casting)

## Geometry (SPICE)
Use SPICE via SpiceyPy for:
- Sun/Earth/Moon positions at UTC
- Body-fixed frames (e.g. IAU_EARTH, IAU_MOON)
- Observer location:
  - Earth surface lon/lat/height OR spacecraft state vector
- Frame transforms for camera pointing and for mapping surface points to lon/lat

## Performance Principles (avoid wasted work)
- Render only the Moon ROI on the detector (compute apparent Moon disk → bounding box)
- Cheap ray–Moon bounding sphere/ellipsoid test first; drop misses immediately
- DEM intersection via iterative “sphere + DEM displacement” (v0), upgradeable to mesh/BVH later
- Earthlight:
  - treat Earth as extended disk
  - avoid per-pixel Earth facet loops by using **tiling / caching**:
    - compute Earth-disk sample directions & Earth-hit mapping for one representative point per tile
    - reuse within tile; refine tiles near limb/horizon as needed
  - early-out when Earth is below local horizon

## Modular Architecture
All user choices live in a single central config (TOML). Suggested modules:

- `config.py` — load/validate config, expose one Config object
- `spice/`
  - `kernels.py` — meta-kernel handling, kernel provenance logging
  - `geometry.py` — states, frames, time conversions (UTC↔ET↔JD)
- `camera.py` — pixel rays, camera pointing/roll, FOV handling
- `moon/`
  - `dem.py` — DEM sampling, gradients → normals
  - `surface.py` — ray hit refine (sphere+DEM); later mesh intersection
  - `brdf.py` — Lambert now, Hapke later (same interface)
- `earth/`
  - `map.py` — albedo sampling
  - `brdf.py` — Lambert now; later ocean/glint/clouds/spectral
  - `extended_source.py` — Earth-disk quadrature + per-tile caching
- `illumination.py` — sunlight + earthlight irradiance at lunar points
- `render.py` — orchestration, masks, tiling, outputs
- `io_fits.py` — FITS writer with int16 scaling + extensive header

## Config (single source of truth)
A `scene.toml` config controls all run-time choices, including:
- UTC timestamp
- observer specification
- camera FOV, image size, roll
- DEM and albedo map paths
- illumination toggles and sampling counts
- shadowing mode
- output paths and metadata tags

## FITS Header Requirements
Must include at least:
- `JD-OBS` as **string** formatted `f15.7`
- UTC string: `DATE-OBS` (ISO8601)
- Observer definition (site or spacecraft)
- Camera: `NAXIS1/2`, `FOV_DEG`, `CRPIX*`, `CDELT*`, pointing definition
- Key geometry scalars: observer–Moon range, Moon phase angle, Earth angular diameter (as seen from Moon centre or representative point), etc.
- Data provenance: DEM/albedo filenames, kernel list / meta-kernel name
- Physics settings: BRDF modes, sample counts, tile size, shadow mode
- Scaling: BSCALE/BZERO and physical units (`BUNIT`)

## Planned Upgrades
1) Spectral/filtered rendering:
- wavelength grid + solar spectrum
- wavelength-dependent BRDFs (Earth + Moon)

2) Moon Hapke photometry:
- global params initially; later spatially varying parameter maps

3) Better shadowing:
- horizon maps or ray casting against a mesh

4) Sensor realism:
- PSF, exposure, noise model (optional)

# Earth Colour Concept Notes

## Scope

These notes describe a practical path from the current scalar Earth albedo
model toward plausible RGB Earth images and movies, without using full
radiative transfer.

The target is not line-by-line physical realism. The target is:

- visually credible Earth colour,
- physically sensible flux handling,
- inputs that can be traced to known geophysical products,
- a design that fits the existing `synthmoon` architecture.

## Core Position

The recommended model is not "take one grayscale Earth image and tint it".

Instead, treat red, green, and blue as three broad bands with their own
effective reflectances. For each Earth patch, compute an RGB reflectance
triplet from surface type, snow/ice state, cloud contribution, and a simple
atmospheric modifier.

This is a three-band reflectance model, not a true radiative transfer model.

## Data Products

Useful products in this framework are mostly not colour images by themselves.
They are control layers.

Examples:

- land-cover class map, e.g. IGBP / MCD12Q1,
- cloud mask or cloud fraction,
- sea-ice concentration,
- land snow / land ice mask,
- optional broadband or multi-band surface albedo / reflectance.

These layers tell us what a pixel is made of, not what display RGB it should be
directly.

## Surface Colour Strategy

Each surface class gets a representative RGB reflectance triplet.

Examples of classes:

- ocean,
- desert,
- bare soil,
- grassland / cropland,
- forest,
- tundra,
- permanent ice / snow,
- cloud.

IGBP-style categories are suitable for this because they are categorical and
stable. Sea ice and land snow / ice should override the base surface category
where present.

This means the renderer works with per-band reflectance values such as:

- `R_surface`,
- `G_surface`,
- `B_surface`.

These are reflectances, not display colours.

## Clouds

Clouds can be handled as a near-neutral bright layer with mild solar tinting if
desired.

Without full radiative transfer, a sensible first approximation is:

- high reflectance in all three bands,
- weak spectral slope,
- coverage given by cloud fraction or cloud mask,
- optional optical-thickness-dependent brightness.

For an MVP, cloud fraction plus a cloud RGB reflectance triplet is enough.

## Snow and Ice

Snow and ice can also be treated as bright, near-neutral reflectors with band
values different from both cloud and desert.

Recommended distinction:

- sea ice,
- land snow,
- permanent land ice.

These should act as overrides or weighted replacements of the underlying
surface, not as unconstrained additive layers.

## Flux Consistency

This is the most important modelling rule.

Do not add surface, cloud, ice, and haze terms as independent positive layers
without limits. That can violate flux conservation.

Instead, compute an effective reflectance per band using mixing rules.

Example:

`A_eff_rgb = (1 - f_cloud) * A_surface_rgb + f_cloud * A_cloud_rgb`

Similarly, snow / ice can replace or mix with the underlying surface according
to coverage fraction.

Then enforce physical bounds:

- `0 <= A_eff_rgb <= 1`

This keeps the colour model interpretable and prevents energy from appearing out
of nowhere.

## Simple Atmospheric Treatment

Atmosphere should not be a pure additive colour wash. Even in a simplified
model, it is better represented as:

- transmission of surface-reflected light, plus
- path radiance added along the line of sight.

Conceptually:

`I_out_rgb = T_rgb * I_surface_rgb + I_path_rgb`

with:

- `0 <= T_rgb <= 1`,
- `I_path_rgb` bounded and linked to the same optical depth that reduces
  transmission.

This keeps the model qualitatively flux-consistent.

## Rayleigh Scattering

Rayleigh is a good candidate for a simple first atmospheric term.

It can be approximated with representative RGB wavelengths, for example:

- red: about 650 nm,
- green: about 550 nm,
- blue: about 450 nm.

Then use the usual wavelength scaling:

- `tau_rayleigh ~ lambda^-4`

Practical use in this project:

- strongest effect near the Earth limb,
- blue limb haze,
- some desaturation / whitening along long slant paths,
- weak global effect away from the limb.

This should be implemented as transmission plus path-radiance, not as an
independent blue layer added on top of everything.

## Aerosols

Aerosols are much harder to do well than Rayleigh.

They are not impossible in a simplified model, but they should probably come
after RGB surfaces, clouds, and Rayleigh are working.

Simple first-order aerosol strategy:

- treat aerosols as a neutral-to-warm haze,
- weaker wavelength dependence than Rayleigh,
- use a power law such as `lambda^-alpha` with modest `alpha`,
- make strongest effects near the limb and in dusty regions if regional data
  exist,
- again, apply as transmission plus path-radiance.

Without an aerosol data product, this should be optional and weak.

## Practical Rendering Stack

For each Earth patch and for each band:

1. determine base surface class,
2. apply snow / ice overrides,
3. mix in cloud contribution by coverage,
4. clamp to a valid effective reflectance,
5. compute direct solar illumination,
6. apply atmospheric transmission,
7. add bounded path radiance,
8. integrate band results into final RGB output.

This is a realistic compromise between current simplicity and a full RT model.

## Recommended MVP

The first useful version should include:

- IGBP-style land cover to assign surface RGB reflectances,
- sea-ice and land snow / ice masks,
- cloud fraction or cloud mask,
- simple ocean glint,
- simple Rayleigh limb haze,
- no aerosol term at first.

That should already produce Earth movies that are much more credible than a
single-band albedo model, while remaining computationally practical.

## Recommended Order Of Work

1. Replace scalar Earth albedo with per-band Earth reflectance.
2. Add land-cover-driven RGB surface classes.
3. Add snow / ice overrides.
4. Add cloud fraction mixing.
5. Add simple Rayleigh transmission plus path-radiance.
6. Add aerosols only if still needed.

## Comparison Criterion

The key comparison against older home-computer work should be:

- does the model remain bounded and interpretable per band,
- does it avoid naive layer addition,
- does it preserve approximate flux logic,
- and does it separate surface, cloud, ice, and atmosphere in a clean way.


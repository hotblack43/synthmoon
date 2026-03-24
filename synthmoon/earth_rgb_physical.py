from __future__ import annotations

from typing import Mapping

import numpy as np


RGBTriplet = tuple[float, float, float]


def parse_rgb_triplet(value: object, default: RGBTriplet) -> RGBTriplet:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(float(np.clip(v, 0.0, 1.0)) for v in value)
    return default


def earth_color_model(earth_cfg: Mapping[str, object]) -> str:
    return str(earth_cfg.get("earth_color_model", "legacy")).strip().lower() or "legacy"


def default_modis_igbp_legacy_table() -> dict[int, RGBTriplet]:
    # Legacy display-oriented RGB table.
    return {
        0: (0.03, 0.06, 0.12),
        1: (0.05, 0.08, 0.04),
        2: (0.06, 0.10, 0.05),
        3: (0.08, 0.11, 0.06),
        4: (0.09, 0.13, 0.07),
        5: (0.07, 0.10, 0.05),
        6: (0.13, 0.12, 0.08),
        7: (0.17, 0.16, 0.11),
        8: (0.16, 0.17, 0.10),
        9: (0.18, 0.18, 0.11),
        10: (0.16, 0.18, 0.10),
        11: (0.10, 0.12, 0.08),
        12: (0.19, 0.18, 0.10),
        13: (0.15, 0.15, 0.15),
        14: (0.18, 0.17, 0.10),
        15: (0.78, 0.82, 0.86),
        16: (0.34, 0.27, 0.18),
        17: (0.18, 0.18, 0.18),
    }


def default_modis_igbp_physical_table() -> dict[int, RGBTriplet]:
    # Approximate visible-band reflectance anchors sampled into camera RGB
    # bands (R~650nm, G~550nm, B~450nm). These are intended as simple
    # physical-ish reflectances rather than display colors.
    return {
        0: (0.02, 0.04, 0.08),   # water
        1: (0.05, 0.07, 0.03),   # evergreen needleleaf forest
        2: (0.05, 0.08, 0.04),   # evergreen broadleaf forest
        3: (0.07, 0.09, 0.05),   # deciduous needleleaf forest
        4: (0.08, 0.11, 0.06),   # deciduous broadleaf forest
        5: (0.07, 0.09, 0.05),   # mixed forests
        6: (0.14, 0.13, 0.09),   # closed shrublands
        7: (0.19, 0.17, 0.11),   # open shrublands
        8: (0.15, 0.16, 0.09),   # woody savannas
        9: (0.17, 0.18, 0.10),   # savannas
        10: (0.14, 0.17, 0.09),  # grasslands
        11: (0.09, 0.11, 0.08),  # permanent wetlands
        12: (0.19, 0.18, 0.11),  # croplands
        13: (0.15, 0.15, 0.15),  # urban / built-up
        14: (0.17, 0.17, 0.10),  # cropland/natural vegetation mosaic
        15: (0.86, 0.88, 0.90),  # permanent snow / ice
        16: (0.31, 0.24, 0.17),  # barren / desert
        17: (0.18, 0.18, 0.18),  # unclassified
    }


def resolve_default_surface_rgb(earth_cfg: Mapping[str, object]) -> RGBTriplet:
    model = earth_color_model(earth_cfg)
    if model == "physical_rgb":
        return parse_rgb_triplet(
            earth_cfg.get("physical_class_rgb_default", earth_cfg.get("class_rgb_default", [0.20, 0.20, 0.20])),
            (0.20, 0.20, 0.20),
        )
    return parse_rgb_triplet(earth_cfg.get("class_rgb_default", [0.20, 0.20, 0.20]), (0.20, 0.20, 0.20))


def resolve_class_rgb_table(earth_cfg: Mapping[str, object]) -> dict[int, RGBTriplet]:
    model = earth_color_model(earth_cfg)
    if model == "physical_rgb":
        preset = str(earth_cfg.get("physical_class_rgb_preset", earth_cfg.get("class_rgb_preset", ""))).strip().lower()
        overrides = earth_cfg.get("physical_class_rgb", {})
        default_table = default_modis_igbp_physical_table
    else:
        preset = str(earth_cfg.get("class_rgb_preset", "")).strip().lower()
        overrides = earth_cfg.get("class_rgb", {})
        default_table = default_modis_igbp_legacy_table

    table: dict[int, RGBTriplet] = {}
    if preset in ("modis_igbp", "igbp", "modis"):
        table.update(default_table())
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            try:
                kk = int(k)
            except (TypeError, ValueError):
                continue
            table[kk] = parse_rgb_triplet(v, table.get(kk, (0.20, 0.20, 0.20)))
    return table


def resolve_cloud_rgb(earth_cfg: Mapping[str, object]) -> RGBTriplet:
    model = earth_color_model(earth_cfg)
    if model == "physical_rgb":
        return parse_rgb_triplet(
            earth_cfg.get("physical_cloud_rgb", earth_cfg.get("cloud_rgb", [0.90, 0.90, 0.90])),
            (0.90, 0.90, 0.90),
        )
    return parse_rgb_triplet(earth_cfg.get("cloud_rgb", [0.78, 0.80, 0.82]), (0.78, 0.80, 0.82))


def resolve_ice_rgb(earth_cfg: Mapping[str, object]) -> RGBTriplet:
    model = earth_color_model(earth_cfg)
    if model == "physical_rgb":
        return parse_rgb_triplet(
            earth_cfg.get("physical_ice_rgb", earth_cfg.get("ice_rgb", [0.84, 0.87, 0.90])),
            (0.84, 0.87, 0.90),
        )
    return parse_rgb_triplet(earth_cfg.get("ice_rgb", [0.78, 0.82, 0.86]), (0.78, 0.82, 0.86))


def simple_brdf_factor(
    earth_cfg: Mapping[str, object],
    *,
    class_id: np.ndarray | None,
    ocean_mask: np.ndarray | None,
    ice_mask: np.ndarray | None,
    mu0: np.ndarray,
    muv: np.ndarray,
) -> np.ndarray:
    if earth_color_model(earth_cfg) != "physical_rgb":
        return np.ones_like(mu0, dtype=np.float64)
    if str(earth_cfg.get("earth_brdf_model", "lambert")).strip().lower() != "simple_kernel":
        return np.ones_like(mu0, dtype=np.float64)

    mu0c = np.clip(np.asarray(mu0, dtype=np.float64), 0.0, 1.0)
    muvc = np.clip(np.asarray(muv, dtype=np.float64), 0.0, 1.0)

    # A mild, empirical non-Lambertian correction that brightens near-nadir /
    # well-lit views and gently dims oblique geometry. The class-dependent
    # strength is the tunable part, not the kernel itself.
    kernel = 0.5 * (mu0c + muvc) - (2.0 / 3.0)
    kernel = np.clip(kernel, -0.35, 0.35)

    out = np.ones_like(mu0c, dtype=np.float64)

    def gain(name: str, default: float) -> float:
        return float(earth_cfg.get(name, default))

    vegetation_classes = {1, 2, 3, 4, 5, 8, 9, 10, 12, 14}
    desert_classes = {6, 7, 16}
    urban_classes = {13}
    wet_classes = {11}

    strength_veg = gain("physical_brdf_strength_vegetation", 0.10)
    strength_desert = gain("physical_brdf_strength_desert", 0.20)
    strength_urban = gain("physical_brdf_strength_urban", 0.08)
    strength_wet = gain("physical_brdf_strength_wetland", 0.06)
    strength_ice = gain("physical_brdf_strength_ice", 0.05)
    strength_land_default = gain("physical_brdf_strength_land_default", 0.10)

    if class_id is not None:
        cls = np.asarray(class_id, dtype=np.float64)
        finite = np.isfinite(cls)
        assigned = np.zeros(cls.shape, dtype=bool)

        def apply_for(classes: set[int], strength: float) -> None:
            nonlocal out, assigned
            mask = finite & np.isin(np.rint(cls).astype(np.int64), np.array(sorted(classes), dtype=np.int64))
            if np.any(mask):
                out[mask] = 1.0 + strength * kernel[mask]
                assigned[mask] = True

        apply_for(vegetation_classes, strength_veg)
        apply_for(desert_classes, strength_desert)
        apply_for(urban_classes, strength_urban)
        apply_for(wet_classes, strength_wet)
        if ice_mask is not None and np.any(ice_mask):
            out[ice_mask] = 1.0 + strength_ice * kernel[ice_mask]
            assigned[ice_mask] = True
        default_land = finite & ~assigned
        if ocean_mask is not None:
            default_land &= ~ocean_mask
        if np.any(default_land):
            out[default_land] = 1.0 + strength_land_default * kernel[default_land]
    elif ice_mask is not None and np.any(ice_mask):
        out[ice_mask] = 1.0 + strength_ice * kernel[ice_mask]

    return np.clip(out, 0.7, 1.3)

from __future__ import annotations

import numpy as np


def ray_sphere_intersect(origins: np.ndarray, dirs: np.ndarray, center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized ray-sphere intersection.
    origins: (N,3), dirs: (N,3) unit, center (3,), radius scalar.
    Returns:
      hit_mask: (N,) bool for rays that hit in front of origin
      t: (N,) distance along ray (km) for nearest positive hit (undefined if miss)
    """
    oc = origins - center[None, :]
    b = 2.0 * np.sum(oc * dirs, axis=1)
    c = np.sum(oc * oc, axis=1) - radius * radius
    disc = b * b - 4.0 * c
    hit = disc >= 0.0
    t = np.full(origins.shape[0], np.nan, dtype=float)
    if not np.any(hit):
        return hit, t
    sqrt_disc = np.sqrt(np.maximum(disc[hit], 0.0))
    b_hit = b[hit]
    # solutions: (-b ± sqrt_disc) / 2
    t0 = (-b_hit - sqrt_disc) / 2.0
    t1 = (-b_hit + sqrt_disc) / 2.0
    t_hit = np.where(t0 > 0, t0, t1)
    hit2 = t_hit > 0
    # update hit mask for only positive
    hit_idx = np.where(hit)[0]
    hit_final_idx = hit_idx[hit2]
    t[hit_final_idx] = t_hit[hit2]
    hit_final = np.zeros_like(hit)
    hit_final[hit_final_idx] = True
    return hit_final, t

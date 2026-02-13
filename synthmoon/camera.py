from __future__ import annotations

import numpy as np


def _safe_normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def camera_basis_from_boresight_and_up(boresight: np.ndarray, up_hint: np.ndarray, roll_deg: float) -> np.ndarray:
    """
    Build a right-handed camera basis in inertial coords.

    Returns a 3x3 matrix R such that:
      d_inertial = R @ d_camera
    where camera axes are:
      +Z: boresight
      +Y: "up"
      +X: right

    roll_deg rotates about +Z in camera coordinates (positive = right-handed).
    """
    z = _safe_normalize(boresight)

    # project up_hint onto plane normal to z
    y0 = up_hint - np.dot(up_hint, z) * z
    if np.linalg.norm(y0) < 1e-12:
        # fallback
        y0 = np.array([0.0, 0.0, 1.0]) - z[2] * z
    y = _safe_normalize(y0)

    x = _safe_normalize(np.cross(y, z))
    y = _safe_normalize(np.cross(z, x))  # re-orthogonalize

    # Apply roll around z: rotate x,y in their plane
    if abs(roll_deg) > 0:
        a = np.deg2rad(roll_deg)
        ca, sa = np.cos(a), np.sin(a)
        x2 = ca * x + sa * y
        y2 = -sa * x + ca * y
        x, y = x2, y2

    R = np.column_stack([x, y, z])  # columns are basis vectors in inertial
    return R


def pixel_rays(nx: int, ny: int, fov_deg: float, R_cam_to_inertial: np.ndarray, roi_bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate ray directions for pixels in ROI bounding box.

    roi_bbox = (x0, x1, y0, y1) inclusive-exclusive: x in [x0,x1), y in [y0,y1)
    Returns:
      ij: (N,2) pixel indices (i,j)
      dirs: (N,3) unit ray directions in inertial frame
    """
    x0, x1, y0, y1 = roi_bbox
    ii, jj = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    ii = ii.ravel()
    jj = jj.ravel()

    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    f = (nx / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)

    x_im = (ii - cx) / f
    y_im = (jj - cy) / f

    # camera coords: +Z forward, +X right, +Y up. Flip y because image j increases down.
    d_cam = np.stack([x_im, -y_im, np.ones_like(x_im)], axis=1)
    d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)

    d_in = (R_cam_to_inertial @ d_cam.T).T
    d_in /= np.linalg.norm(d_in, axis=1, keepdims=True)

    ij = np.stack([ii, jj], axis=1).astype(int)
    return ij, d_in


def moon_roi_bbox(nx: int, ny: int, boresight: np.ndarray, moon_ang_radius_deg: float, fov_deg: float, margin_px: int = 8) -> tuple[int, int, int, int]:
    """
    For pointing at moon center, the moon center projects to image center.
    We compute an approximate bounding box for the moon disk in pixels.
    """
    # pixel scale in deg/px
    px_scale = fov_deg / nx
    r_px = moon_ang_radius_deg / px_scale
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    r = int(np.ceil(r_px)) + margin_px

    x0 = max(0, int(np.floor(cx - r)))
    x1 = min(nx, int(np.ceil(cx + r)))
    y0 = max(0, int(np.floor(cy - r)))
    y1 = min(ny, int(np.ceil(cy + r)))
    return (x0, x1, y0, y1)

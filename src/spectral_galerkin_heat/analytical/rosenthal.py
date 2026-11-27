"""Rosenthal quasi-steady solution for a moving point source, with wall images.

Reference: D. Rosenthal, The theory of moving sources of heat and its
application to metal treatments, Transactions of the ASME 68 (1946) 849-866.

In the laser frame (xi = x - x_laser, R the distance to the source):

    T - T0 = A P / (2 pi k) * exp[-v (xi + R) / (2 alpha)] / R

for a source moving in +x at speed v on the surface of a semi-infinite body.
"""

import numpy as np


def rosenthal_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    rho: float,
    k: float,
    Cp: float,
    T0: float,
    A: float,
    P: float,
    Ly: float,
    Lz: float,
    x_laser: float,
    y_laser: float,
    v: float,
    r_min: float = 1e-9,
    images: bool = True,
) -> np.ndarray:
    """Rosenthal temperature field on the grid ``x, y, z``.

    Parameters
    ----------
    x, y, z : 1-D arrays
        Evaluation coordinates (m), in the domain frame.
    rho, k, Cp, T0 : float
        Density, conductivity, heat capacity, ambient temperature.
    A, P : float
        Absorptivity and laser power (W).
    Ly, Lz : float
        Domain width and height (m), used to place the image sources.
    x_laser, y_laser : float
        Source position (m) on the top face ``z = Lz``.
    v : float
        Laser speed along +x (m/s).
    r_min : float
        Distance floor (m) that keeps the source point finite.
    images : bool, default True
        Add images across the scan-parallel walls ``y = 0``, ``y = Ly`` and
        the bottom ``z = 0``. The quasi-steady solution has no image across the
        x walls (the source is taken to have travelled from x = -inf).

    Returns
    -------
    ndarray, shape (z.size, y.size, x.size), float64
    """
    x, y, z = (np.asarray(c, dtype=np.float64) for c in (x, y, z))
    alpha = k / (rho * Cp)

    xi = (x - x_laser)[None, None, :]
    eta = (y - y_laser)[None, :, None]
    depth = (Lz - z)[:, None, None]

    def kernel(eta_d, depth_d):
        R = np.sqrt(xi**2 + eta_d**2 + depth_d**2)
        R = np.maximum(R, r_min)
        return np.exp(-v / (2 * alpha) * (xi + R)) / R

    T = kernel(eta, depth)
    if images:
        T += kernel(eta + 2 * y_laser, depth)            # y = 0
        T += kernel(eta - 2 * (Ly - y_laser), depth)     # y = Ly
        T += kernel(eta, 2.0 * Lz - depth)               # z = 0
    return T * A * P / (2 * np.pi * k) + T0

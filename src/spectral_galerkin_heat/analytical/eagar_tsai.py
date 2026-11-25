"""Eagar-Tsai solution for a moving Gaussian heat source, with wall images.

Reference: T. W. Eagar, N. S. Tsai, Temperature fields produced by traveling
distributed heat sources, Welding Journal 62 (12) (1983) 346s-355s.

In the laser frame (xi = x - x_laser, eta = y - y_laser, depth d = Lz - z):

    T - T0 = A P / [pi rho Cp sqrt(pi alpha)]
             * int_0^t  exp[-((xi + v tau)^2 + eta^2) / (4 alpha tau + sigma^2)
                             - d^2 / (4 alpha tau)]
                        / [sqrt(tau) (4 alpha tau + sigma^2)]  d tau

with sigma^2 = r_b^2 / 2, where r_b is the 1/e^2 beam radius used by the solver,
``I(r) = 2 A P / (pi r_b^2) exp(-2 r^2 / r_b^2)``.

The integrand is a product of one factor per axis, so the insulated walls of
the cuboid are imposed exactly by replacing each factor with its 1-D image
series (reflections across both walls of that axis), truncated at
``image_order`` reflections.

Time-stepped source. Passing the solver's ``dt`` here: the source emitting at
time t' sits where the laser was at the start of its step.
"""

import numpy as np
from scipy.special import erf


def _axis_factor(X, s, L, D, c, sigma_sq, order, images, clip):
    """In-plane factor of the kernel along one axis, with its image series.

    ``X`` are the evaluation coordinates, ``s`` the source position, ``L`` the
    domain length on this axis, ``D = c + sigma_sq`` with ``c = 4 alpha tau``.
    With ``clip``, the Gaussian beam is restricted to ``[0, L]`` before imaging.
    """
    def term(Xe):
        g = np.exp(-(Xe - s)**2 / D)
        if clip:
            mu = (s * c + Xe * sigma_sq) / D
            w = np.sqrt(sigma_sq * c / D)
            g = g * 0.5 * (erf((L - mu) / w) + erf(mu / w))
        return g

    if not images:
        return term(X)
    m = np.arange(-order, order + 1)[:, None]
    return (term(X[None, :] - 2 * m * L) + term(2 * m * L - X[None, :])).sum(axis=0)


def eagar_tsai_field(
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
    r_b: float,
    Lx: float,
    Ly: float,
    Lz: float,
    x_laser: float,
    y_laser: float,
    v: float,
    t_total: float,
    dt: float | None = None,
    n_tau: int = 400,
    tau_min: float = 1e-10,
    images: bool = True,
    image_order: int = 1,
    clip_source: bool = False,
) -> np.ndarray:
    """Eagar-Tsai temperature field on the grid ``x, y, z``.

    The field is evaluated at the instant the laser sits at
    ``(x_laser, y_laser)`` on the top face ``z = Lz``, after travelling at
    constant signed velocity ``v`` along x for ``t_total`` seconds.

    Parameters
    ----------
    x, y, z : 1-D arrays
        Evaluation coordinates (m), in the domain frame ``[0, Lx] x [0, Ly] x
        [0, Lz]``.
    rho, k, Cp, T0 : float
        Density, conductivity, heat capacity, initial and ambient temperature.
    A, P, r_b : float
        Absorptivity, laser power (W) and 1/e^2 beam radius (m).
    Lx, Ly, Lz : float
        Domain size (m), used to place the image sources.
    x_laser, y_laser : float
        Laser position (m) at the evaluation instant.
    v : float
        Signed laser velocity along x (m/s).
    t_total : float
        Travel time (s), the upper bound of the tau integral.
    dt : float, optional
        Solver time step (s). When given, the source is held at its
        start-of-step position over each step, as in the solver.
    n_tau, tau_min : int, float
        Number of geometric quadrature points and lower cut-off of the tau
        integral.
    images : bool, default True
        Impose the insulated walls with image sources: on each axis, the
        reflections across both walls up to ``image_order``, multiplied across
        the three axes (edge and corner images included). The heated top face
        is a wall of the z series. False returns the bare semi-infinite
        solution.
    image_order : int, default 1
        Number of reflections kept on each side of each axis.
    clip_source : bool, default False
        Restrict the Gaussian beam to the top face ``[0, Lx] x [0, Ly]``, as
        the solver does when the beam overlaps a side wall.

    Returns
    -------
    ndarray, shape (z.size, y.size, x.size), float64
    """
    x, y, z = (np.asarray(c, dtype=np.float64) for c in (x, y, z))
    alpha = k / (rho * Cp)

    tau_pts = np.geomspace(tau_min, t_total, n_tau)
    dtau = np.diff(tau_pts)

    depth = Lz - z
    sigma_sq = r_b**2 / 2.0
    C = A * P / (np.pi * rho * Cp * np.sqrt(np.pi * alpha))
    p = np.arange(-image_order, image_order + 1)[:, None] if images else np.zeros((1, 1))

    T_field = np.zeros((z.size, y.size, x.size))
    for i in range(n_tau - 1):
        tau = 0.5 * (tau_pts[i] + tau_pts[i + 1])
        c = 4.0 * alpha * tau
        D = c + sigma_sq

        # Distance travelled by the laser since the source emitted: v * lag.
        # Continuous motion: lag = tau. Stepped: the emitting source sits at
        # its start-of-step position, so it lags the laser by more than tau.
        lag = tau if dt is None else t_total - dt * np.floor((t_total - tau) / dt)
        x_src = x_laser - v * lag

        fx = _axis_factor(x, x_src, Lx, D, c, sigma_sq, image_order, images, clip_source)
        fy = _axis_factor(y, y_laser, Ly, D, c, sigma_sq, image_order, images, clip_source)
        # Source on the top face: its images sit at depths 2 p Lz.
        fz = np.exp(-(depth[None, :] + 2 * p * Lz)**2 / c).sum(axis=0)

        scalar = dtau[i] / (np.sqrt(tau) * D)
        T_field += (scalar * fz)[:, None, None] * (fy[:, None] * fx[None, :])[None, :, :]

    return T_field * C + T0

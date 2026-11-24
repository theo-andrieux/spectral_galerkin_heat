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

Time-stepped source. Passing the solver's ``dt`` here: the source emitting at 
time t' sits where the laser was at the start of its step.
"""

import numpy as np


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
        Add five image sources (reflections across x = 0, x = Lx, y = 0,
        y = Ly and the bottom z = 0) so the field satisfies the insulated walls.
        The heated top face needs no image. False returns the bare
        semi-infinite solution.

    Returns
    -------
    ndarray, shape (z.size, y.size, x.size), float64
    """
    x, y, z = (np.asarray(c, dtype=np.float64) for c in (x, y, z))
    alpha = k / (rho * Cp)

    tau_pts = np.geomspace(tau_min, t_total, n_tau)
    dtau = np.diff(tau_pts)

    xi = x - x_laser
    eta = y - y_laser
    depth = Lz - z
    eta_img_y0 = eta + 2 * y_laser
    eta_img_yLy = eta - 2 * (Ly - y_laser)
    depth_img_z0 = 2.0 * Lz - depth

    sigma_sq = r_b**2 / 2.0
    C = A * P / (np.pi * rho * Cp * np.sqrt(np.pi * alpha))

    T_field = np.zeros((z.size, y.size, x.size))
    buf = np.empty_like(T_field)

    for i in range(n_tau - 1):
        tau = 0.5 * (tau_pts[i] + tau_pts[i + 1])
        denom_xy = 4.0 * alpha * tau + sigma_sq
        denom_z = 4.0 * alpha * tau
        scalar = dtau[i] / (np.sqrt(tau) * denom_xy)

        # Distance travelled by the laser since the source emitted: v * lag.
        # Continuous motion: lag = tau. Stepped: the emitting source sits at
        # its start-of-step position, so it lags the laser by more than tau.
        lag = tau if dt is None else t_total - dt * np.floor((t_total - tau) / dt)
        exp_x = np.exp(-(xi + v * lag)**2 / denom_xy)
        exp_y = np.exp(-eta**2 / denom_xy)
        exp_z = np.exp(-depth**2 / denom_z)
        f_xy = exp_y[:, None] * exp_x[None, :]

        if images:
            # Side walls. A reflection across an x wall reverses the velocity.
            exp_x_img_x0 = np.exp(-(xi + 2 * x_laser - v * lag)**2 / denom_xy)
            exp_x_img_xLx = np.exp(-(xi + 2 * x_laser - 2 * Lx - v * lag)**2 / denom_xy)
            f_xy_side = (exp_y[:, None] * (exp_x_img_x0 + exp_x_img_xLx)[None, :]
                         + (np.exp(-eta_img_y0**2 / denom_xy)
                            + np.exp(-eta_img_yLy**2 / denom_xy))[:, None] * exp_x[None, :])
            np.multiply(exp_z[:, None, None], (f_xy + f_xy_side)[None, :, :], out=buf)
            # Bottom wall: same in-plane profile, mirrored depth.
            exp_z_img = np.exp(-depth_img_z0**2 / denom_z)
            buf += exp_z_img[:, None, None] * f_xy[None, :, :]
        else:
            np.multiply(exp_z[:, None, None], f_xy[None, :, :], out=buf)

        buf *= scalar
        T_field += buf

    return T_field * C + T0

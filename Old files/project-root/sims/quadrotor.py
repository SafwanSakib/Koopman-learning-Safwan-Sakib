"""Planar (3-DOF) quadrotor — Track B §1.1, Research Plan §7.1.

Matches Folkestad et al. (2020)'s benchmark family; collision-avoidance safe
sets are the natural CBF story here (Research Plan §7.1).

Planar quadrotor state x = [x, z, phi, x_dot, z_dot, phi_dot] (horizontal
position, altitude, pitch angle, and their rates). Inputs u = [u1, u2] are
the two rotor thrusts.

    x_ddot   = -(u1+u2)*sin(phi)/m
    z_ddot   =  (u1+u2)*cos(phi)/m - g
    phi_ddot =  (u2-u1)*r/I

This is the standard planar-quadrotor parameterization used throughout the
Koopman-CBF literature (Folkestad et al. 2020 uses a 3D variant; this planar
version is the lower-difficulty starting point per Research Plan §7.1's own
"Medium-High" difficulty note — upgrade to full 3D only if time allows).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

G = 9.81


def dynamics(t: float, x: np.ndarray, u: np.ndarray,
             m: float = 1.0, I: float = 0.01, r: float = 0.25) -> np.ndarray:
    _, _, phi, x_dot, z_dot, phi_dot = x
    u1, u2 = u
    x_ddot = -(u1 + u2) * np.sin(phi) / m
    z_ddot = (u1 + u2) * np.cos(phi) / m - G
    phi_ddot = (u2 - u1) * r / I
    return np.array([x_dot, z_dot, phi_dot, x_ddot, z_ddot, phi_ddot])


def simulate(x0: np.ndarray, u_fn, t_span: tuple[float, float], dt: float,
             m: float = 1.0, I: float = 0.01, r: float = 0.25) -> dict:
    t_eval = np.arange(t_span[0], t_span[1], dt)

    def rhs(t, x):
        return dynamics(t, x, u_fn(t), m=m, I=I, r=r)

    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    u_vals = np.array([u_fn(t) for t in t_eval])
    return {"t": sol.t, "x": sol.y.T, "u": u_vals}


def generate_pe_trajectory(x0: np.ndarray, t_span: tuple[float, float], dt: float,
                            seed: int = 0, excitation: str = "prbs", **kwargs) -> dict:
    raise NotImplementedError("Week 1-2: implement PE input generation")


def safe_set(x: np.ndarray, obstacles: list[tuple[float, float, float]]) -> np.ndarray:
    """Collision-avoidance h(x): min over obstacles of (distance to obstacle
    center - obstacle radius - vehicle radius). `obstacles` is a list of
    (cx, cz, radius) tuples.

    TODO (Week 6-10, quadrotor onboarding): same min()-smoothness caveat as
    cartpole/cstr applies, doubly so here since there may be multiple
    obstacles simultaneously active.
    """
    raise NotImplementedError("Week 6-10: finalize h(x) formulation, see docstring")

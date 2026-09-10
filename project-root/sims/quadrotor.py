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
                            seed: int = 0, excitation: str = "prbs",
                            hold_time: float = 0.3, amplitude: float = 0.3,
                            m: float = 1.0, I: float = 0.01, r: float = 0.25) -> dict:
    """Generate a trajectory under a PRBS input PERTURBING each rotor
    around its hover thrust (m*g/2 each) -- like sims/cstr.py, excitation
    stays local around a physically sensible operating point rather than
    swinging near zero, which would just make the vehicle fall."""
    rng = np.random.default_rng(seed)
    t_eval = np.arange(t_span[0], t_span[1], dt)
    hover_thrust = m * G / 2

    if excitation == "prbs":
        switch_every = max(1, int(round(hold_time / dt)))
        n_switches = len(t_eval) // switch_every + 2
        levels_1 = hover_thrust + rng.choice([-amplitude, amplitude], size=n_switches)
        levels_2 = hover_thrust + rng.choice([-amplitude, amplitude], size=n_switches)
        u1_vals = np.repeat(levels_1, switch_every)[: len(t_eval)]
        u2_vals = np.repeat(levels_2, switch_every)[: len(t_eval)]
    else:
        raise ValueError(f"Unknown excitation scheme: {excitation}")

    def u_fn(t):
        idx = min(int(round((t - t_span[0]) / dt)), len(u1_vals) - 1)
        return np.array([u1_vals[idx], u2_vals[idx]])

    result = simulate(x0, u_fn, t_span, dt, m=m, I=I, r=r)
    result["u"] = np.stack([u1_vals, u2_vals], axis=1)
    return result

def safe_set(x: np.ndarray, obstacles: list[tuple[float, float, float]]) -> dict:
    """RESOLVED 2026-09-10 -- same resolution as sims/cartpole.py's
    safe_set, doubly important here since obstacle count varies per scene:
    return ONE SEPARATE barrier function PER obstacle (distance to that
    obstacle's center - its radius - vehicle radius), never min()-combined
    across obstacles. See sims/cartpole.py's safe_set docstring for the
    full rationale (notation_assumptions.tex, h(x) = e_j^T Phi(x)
    single-coordinate requirement).

    Consequence for models/autoencoder.py: h_coordinates must have one
    entry per obstacle in the SPECIFIC SCENE being trained/evaluated on --
    the lift_dim and h_coordinates list are therefore scene-dependent for
    this benchmark, unlike cart-pole/CSTR where the bound count is fixed.
    Document the obstacle count used per experiment config
    (experiments/configs/) so results are reproducible.

    Parameters
    ----------
    obstacles : list of (cx, cz, radius) tuples.

    Returns
    -------
    dict with keys "obstacle_0", "obstacle_1", ..., each h(x) >= 0 inside
    the safe region for that obstacle.
    """
    px, pz = x[0], x[1]
    vehicle_radius = 0.15  # placeholder, TODO: source from a shared config
    return {
        f"obstacle_{i}": float(np.hypot(px - cx, pz - cz) - r - vehicle_radius)
        for i, (cx, cz, r) in enumerate(obstacles)
    }

"""Inverted pendulum / cart-pole — Track B §1.1, Research Plan §7.1.

Natural CBF safe set: pole angle and/or cart position bounds.

State x = [p, p_dot, theta, theta_dot] (cart position, cart velocity, pole
angle from upright, pole angular velocity). Standard cart-pole equations of
motion (control-affine in u, the horizontal force on the cart):

    theta_ddot = (g*sin(theta) - cos(theta)*(u + m_p*l*theta_dot^2*sin(theta))/(m_c+m_p))
                 / (l*(4/3 - m_p*cos(theta)^2/(m_c+m_p)))
    p_ddot     = (u + m_p*l*(theta_dot^2*sin(theta) - theta_ddot*cos(theta))) / (m_c+m_p)

This is the textbook (Florian 2007 / Barto-Sutton-Anderson) parameterization.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

G = 9.81


def dynamics(t: float, x: np.ndarray, u: float,
             m_c: float = 1.0, m_p: float = 0.1, l: float = 0.5) -> np.ndarray:
    p, p_dot, theta, theta_dot = x
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    total_mass = m_c + m_p

    theta_ddot = (
        G * sin_t - cos_t * (u + m_p * l * theta_dot ** 2 * sin_t) / total_mass
    ) / (l * (4.0 / 3.0 - m_p * cos_t ** 2 / total_mass))

    p_ddot = (u + m_p * l * (theta_dot ** 2 * sin_t - theta_ddot * cos_t)) / total_mass

    return np.array([p_dot, p_ddot, theta_dot, theta_ddot])


def simulate(x0: np.ndarray, u_fn, t_span: tuple[float, float], dt: float,
             m_c: float = 1.0, m_p: float = 0.1, l: float = 0.5) -> dict:
    t_eval = np.arange(t_span[0], t_span[1], dt)

    def rhs(t, x):
        return dynamics(t, x, u_fn(t), m_c=m_c, m_p=m_p, l=l)

    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    u_vals = np.array([u_fn(t) for t in t_eval])
    return {"t": sol.t, "x": sol.y.T, "u": u_vals}


def generate_pe_trajectory(x0: np.ndarray, t_span: tuple[float, float], dt: float,
                            seed: int = 0, excitation: str = "prbs",
                            hold_time: float = 0.3, amplitude: float = 2.0,
                            m_c: float = 1.0, m_p: float = 0.1, l: float = 0.5) -> dict:
    """Generate a trajectory under a PRBS input, same scheme as
    sims/van_der_pol.py (see that module for the rationale)."""
    rng = np.random.default_rng(seed)
    t_eval = np.arange(t_span[0], t_span[1], dt)

    if excitation == "prbs":
        switch_every = max(1, int(round(hold_time / dt)))
        n_switches = len(t_eval) // switch_every + 2
        levels = rng.choice([-amplitude, amplitude], size=n_switches)
        u_vals = np.repeat(levels, switch_every)[: len(t_eval)]
    else:
        raise ValueError(f"Unknown excitation scheme: {excitation}")

    def u_fn(t):
        idx = min(int(round((t - t_span[0]) / dt)), len(u_vals) - 1)
        return u_vals[idx]

    result = simulate(x0, u_fn, t_span, dt, m_c=m_c, m_p=m_p, l=l)
    result["u"] = u_vals
    return result


def safe_set(x: np.ndarray, theta_max: float = 0.5, p_max: float = 2.4) -> dict:
    """RESOLVED 2026-09-10 per the frozen Track A theory note
    (notation_assumptions.tex: h(x) = e_j^T Phi(x) must be a SINGLE linear
    coordinate). The angle bound and position bound are returned as TWO
    SEPARATE barrier functions, NOT combined via min() -- each gets its own
    embedded coordinate in models/autoencoder.py's h_coordinates list and
    its own hard first-step CBF constraint (controllers/cbf_constraint.py).
    Theorem 2 applies to each independently, giving
    h_angle(x_k) >= -eta_1/gamma AND h_position(x_k) >= -eta_1/gamma
    simultaneously -- the AND-of-bounds semantics a min() was trying to
    express, without the non-embeddable nonlinearity.

    Returns
    -------
    dict with keys "angle" and "position", each h(x) = bound - |value|
    (>= 0 inside the safe region).
    """
    _, _, theta, _ = x
    p = x[0]
    return {"angle": theta_max - abs(theta), "position": p_max - abs(p)}

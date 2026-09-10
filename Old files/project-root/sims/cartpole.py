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
                            seed: int = 0, excitation: str = "prbs", **model_kwargs) -> dict:
    raise NotImplementedError("Week 1-2: implement PE input generation")


def safe_set(x: np.ndarray, theta_max: float = 0.5, p_max: float = 2.4) -> np.ndarray:
    """h(x) = min(theta_max - |theta|, p_max - |p|) style combined bound.

    TODO (Week 2): decide whether to use a smooth (differentiable) surrogate
    for the min() so h is nicer for the Koopman-embedding trick
    (Theoretical_Background_Full.md §6.4) — a hard min complicates embedding
    h as a single linear coordinate of z. A softmin or two separate CBFs
    (one per bound) may be preferable; revisit once models/autoencoder.py's
    h_coordinate contract is implemented.
    """
    raise NotImplementedError("Week 2: finalize h(x) formulation, see docstring")

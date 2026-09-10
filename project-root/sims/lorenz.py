"""Lorenz system — Track B §1.1, Research Plan §7.1.

Chaotic stress-test for the lifting/prediction-accuracy claims. Per Research
Plan §7.1: "mainly for a robustness appendix, not core benchmark" — no
safety/CBF story needed here (no safe_set function), and no control input in
the classical formulation (autonomous system; optionally add a forcing term
if the experiment plan wants a controlled variant later).

    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x*y - beta*z

Classic chaotic regime: sigma=10, rho=28, beta=8/3.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def dynamics(t: float, x: np.ndarray, u: float = 0.0,
             sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    x1, x2, x3 = x
    return np.array([
        sigma * (x2 - x1),
        x1 * (rho - x3) - x2 + u,  # optional additive forcing u, defaults to autonomous
        x1 * x2 - beta * x3,
    ])


def simulate(x0: np.ndarray, t_span: tuple[float, float], dt: float, u_fn=None,
             sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> dict:
    if u_fn is None:
        u_fn = lambda t: 0.0  # noqa: E731 — autonomous by default
    t_eval = np.arange(t_span[0], t_span[1], dt)

    def rhs(t, x):
        return dynamics(t, x, u_fn(t), sigma=sigma, rho=rho, beta=beta)

    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, rtol=1e-9, atol=1e-11)
    u_vals = np.array([u_fn(t) for t in t_eval])
    return {"t": sol.t, "x": sol.y.T, "u": u_vals}


def generate_pe_trajectory(x0: np.ndarray, t_span: tuple[float, float], dt: float,
                            seed: int = 0, excitation: str = "prbs",
                            hold_time: float = 0.1, amplitude: float = 2.0,
                            sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> dict:
    """Generate a trajectory under a PRBS forcing term (via the optional
    additive u in dynamics()). Lorenz is autonomous in its classical form
    and used only as a prediction-accuracy stress test (Research Plan
    Sec.7.1), not a control benchmark -- this exists mainly for interface
    consistency with the other four simulators, and in case a controlled
    variant is wanted later."""
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

    result = simulate(x0, t_span, dt, u_fn=u_fn, sigma=sigma, rho=rho, beta=beta)
    result["u"] = u_vals
    return result

# Intentionally no safe_set() here — see module docstring.

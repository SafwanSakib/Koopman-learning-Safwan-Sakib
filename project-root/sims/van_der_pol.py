"""Van der Pol oscillator — Track B §1.1, Research Plan §7.1.

Classic Koopman benchmark with a well-understood limit cycle, used as the
first end-to-end pipeline target (Track B §3, Weeks 4-6).

State x = [x1, x2]. Controlled (forced) Van der Pol:
    x1_dot = x2
    x2_dot = mu * (1 - x1^2) * x2 - x1 + u

Sanity check (Track B §1.1): for u=0 and mu > 0, trajectories converge to a
limit cycle; for small mu the amplitude is close to 2 (standard result) —
verify this numerically before trusting the simulator for anything else.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def dynamics(t: float, x: np.ndarray, u: float, mu: float = 1.0) -> np.ndarray:
    x1, x2 = x
    return np.array([x2, mu * (1 - x1 ** 2) * x2 - x1 + u])


def simulate(x0: np.ndarray, u_fn, t_span: tuple[float, float], dt: float,
             mu: float = 1.0) -> dict:
    """Simulate forward with a (possibly time-varying) input function u_fn(t).

    Returns dict with keys: t, x (shape (T, 2)), u (shape (T,)).
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)

    def rhs(t, x):
        return dynamics(t, x, u_fn(t), mu=mu)

    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    u_vals = np.array([u_fn(t) for t in t_eval])
    return {"t": sol.t, "x": sol.y.T, "u": u_vals}


def generate_pe_trajectory(x0: np.ndarray, t_span: tuple[float, float], dt: float,
                            mu: float = 1.0, seed: int = 0, excitation: str = "prbs",
                            hold_time: float = 0.5, amplitude: float = 1.0) -> dict:
    """Generate a trajectory under a persistently-exciting input scheme.

    excitation="prbs": switches u randomly between -amplitude and +amplitude,
    holding each value for `hold_time` seconds before the next random switch.
    This is the standard, simplest PE input used throughout the DeePC
    literature (Theoretical_Background_Full.md Sec.2.3).
    """
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

    result = simulate(x0, u_fn, t_span, dt, mu=mu)
    result["u"] = u_vals  # overwrite with the exact PRBS array (avoids float-index rounding drift)
    return result


def safe_set(x: np.ndarray) -> np.ndarray:
    """h(x) for Van der Pol. Per Research Plan §7.1, Van der Pol has no
    natural safety constraint in the base benchmark set (cart-pole/CSTR/
    quadrotor carry the safety-constraint stories) — if a bound is needed
    for a CBF smoke test, use a simple amplitude bound as a placeholder.

    TODO: confirm during Week 4-6 whether Van der Pol needs a real h(x) for
    the first end-to-end pipeline milestone, or whether the first CBF
    validation should happen directly on cart-pole instead.
    """
    raise NotImplementedError("Week 4: define or defer, see docstring")

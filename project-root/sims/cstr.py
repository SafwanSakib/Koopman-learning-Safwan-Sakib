"""Continuous stirred-tank reactor (CSTR) — Track B §1.1, Research Plan §7.1.

Natural safety constraints: temperature/concentration bounds. Matches prior
physics-informed Koopman work (arXiv:2503.18787) for comparability, per
Research Plan §7.1.

Standard single-exothermic-reaction, non-isothermal CSTR model (state:
concentration C_A, reactor temperature T; input: coolant temperature or
coolant flow, here taken as coolant temperature T_c for simplicity):

    C_A_dot = (q/V)(C_Af - C_A) - k0*exp(-E/(R*T))*C_A
    T_dot   = (q/V)(Tf - T) + (-dH/(rho*Cp))*k0*exp(-E/(R*T))*C_A
              + (UA/(V*rho*Cp))*(T_c - T)

This is the classic Seborg/Bequette CSTR benchmark parameterization; exact
constants are placeholders and should be swapped for the specific case study
constants used in the physics-informed Koopman eNMPC reference paper once
that's re-read in full (Research Plan §4.4 caveat: re-verify against primary
source before finalizing).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# Placeholder parameters (Seborg-style CSTR). TODO: replace with the exact
# constants from the reference comparability paper (arXiv:2503.18787) during
# Week 6-10 CSTR onboarding, and cite the source in this docstring.
DEFAULTS = dict(
    q=100.0,       # volumetric flow rate (L/min)
    V=100.0,       # reactor volume (L)
    C_Af=1.0,      # feed concentration (mol/L)
    Tf=350.0,      # feed temperature (K)
    k0=7.2e10,     # pre-exponential factor (1/min)
    E_over_R=8750.0,  # activation energy / gas constant (K)
    dH=-5e4,       # heat of reaction (J/mol)
    rho=1000.0,    # density (g/L)
    Cp=0.239,      # heat capacity (J/(g K))
    UA=5e4,        # heat transfer coefficient * area (J/(min K))
)


def dynamics(t: float, x: np.ndarray, u: float, params: dict | None = None) -> np.ndarray:
    """u is coolant temperature T_c (K). x = [C_A, T]."""
    p = {**DEFAULTS, **(params or {})}
    C_A, T = x
    k = p["k0"] * np.exp(-p["E_over_R"] / T)
    C_A_dot = (p["q"] / p["V"]) * (p["C_Af"] - C_A) - k * C_A
    T_dot = (
        (p["q"] / p["V"]) * (p["Tf"] - T)
        + (-p["dH"] / (p["rho"] * p["Cp"])) * k * C_A
        + (p["UA"] / (p["V"] * p["rho"] * p["Cp"])) * (u - T)
    )
    return np.array([C_A_dot, T_dot])


def simulate(x0: np.ndarray, u_fn, t_span: tuple[float, float], dt: float,
             params: dict | None = None) -> dict:
    t_eval = np.arange(t_span[0], t_span[1], dt)

    def rhs(t, x):
        return dynamics(t, x, u_fn(t), params=params)

    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    u_vals = np.array([u_fn(t) for t in t_eval])
    return {"t": sol.t, "x": sol.y.T, "u": u_vals}

def compute_steady_state(T_c: float, params: dict | None = None,
                          T_bounds: tuple[float, float] = (280.0, 450.0)) -> np.ndarray:
    """Find the steady-state (C_A, T) for a constant coolant temperature
    T_c. Standard chemical-engineering technique (Seborg et al.): the
    steady-state mass balance gives C_A directly as a function of T, so
    substituting into the energy balance reduces the problem to a SINGLE
    equation in T alone, solved by bisection (brentq) rather than a 2D
    Newton solve -- far more robust for this system, which is known to
    have multiple steady states (an S-shaped ignition/extinction curve)
    in certain T_c ranges. brentq requires the residual to change sign
    across T_bounds; if it doesn't (e.g. T_c lands exactly in the
    multiplicity region where 3 roots exist within a narrower window),
    narrow T_bounds to isolate the branch of interest.
    """
    from scipy.optimize import brentq

    p = {**DEFAULTS, **(params or {})}

    def energy_residual(T):
        k = p["k0"] * np.exp(-p["E_over_R"] / T)
        C_A = (p["q"] / p["V"]) * p["C_Af"] / (p["q"] / p["V"] + k)
        return (
            (p["q"] / p["V"]) * (p["Tf"] - T)
            + (-p["dH"] / (p["rho"] * p["Cp"])) * k * C_A
            + (p["UA"] / (p["V"] * p["rho"] * p["Cp"])) * (T_c - T)
        )

    T_ss = brentq(energy_residual, T_bounds[0], T_bounds[1])
    k = p["k0"] * np.exp(-p["E_over_R"] / T_ss)
    C_A_ss = (p["q"] / p["V"]) * p["C_Af"] / (p["q"] / p["V"] + k)
    return np.array([C_A_ss, T_ss])

def generate_pe_trajectory(x0: np.ndarray, t_span: tuple[float, float], dt: float,
                            seed: int = 0, excitation: str = "prbs",
                            hold_time: float = 1.0, u_nominal: float = 300.0,
                            amplitude: float = 5.0, params: dict | None = None) -> dict:
    """Generate a trajectory under a PRBS input PERTURBING a nominal coolant
    temperature (not swinging between two extremes like the other
    benchmarks) -- CSTR is sensitive to operating region, so excitation
    stays local around u_nominal."""
    rng = np.random.default_rng(seed)
    t_eval = np.arange(t_span[0], t_span[1], dt)

    if excitation == "prbs":
        switch_every = max(1, int(round(hold_time / dt)))
        n_switches = len(t_eval) // switch_every + 2
        levels = u_nominal + rng.choice([-amplitude, amplitude], size=n_switches)
        u_vals = np.repeat(levels, switch_every)[: len(t_eval)]
    else:
        raise ValueError(f"Unknown excitation scheme: {excitation}")

    def u_fn(t):
        idx = min(int(round((t - t_span[0]) / dt)), len(u_vals) - 1)
        return u_vals[idx]

    result = simulate(x0, u_fn, t_span, dt, params=params)
    result["u"] = u_vals
    return result

def safe_set(x: np.ndarray, T_max: float = 400.0, C_A_min: float = 0.1) -> dict:
    """RESOLVED 2026-09-10 -- same resolution as sims/cartpole.py's
    safe_set: the temperature ceiling and concentration floor are returned
    as TWO SEPARATE barrier functions (each a single linear coordinate of
    the lifted state once embedded), not min()-combined. See
    sims/cartpole.py's safe_set docstring for the full rationale
    (notation_assumptions.tex, h(x) = e_j^T Phi(x) single-coordinate
    requirement).

    Returns
    -------
    dict with keys "temperature" and "concentration", each h(x) >= 0
    inside the safe region.
    """
    C_A, T = x
    return {"temperature": T_max - T, "concentration": C_A - C_A_min}

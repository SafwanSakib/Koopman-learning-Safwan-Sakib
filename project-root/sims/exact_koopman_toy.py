"""Toy nonlinear system with a known EXACT finite-dimensional Koopman
embedding -- added 2026-09-05 per Track C's literature lock
(Consolidated_Findings_TrackA_TrackB_Revisions.md, Part 3.2 / Part 5 item 6).

Purpose: Theorem 1 is now framed as extending Shang, Cortes, Zheng (2024)'s
PE-transfer result for EXACT Koopman embeddings to the case of a LEARNED,
approximate (physics-informed) lifting with error epsilon_phys > 0. This
module provides the exact-embedding "ground truth" limit (epsilon_phys = 0)
that experiments/pe_transfer_sanity_check.py perturbs away from -- i.e. this
is the anchor point Shang et al. already proved, not something we're
claiming as novel.

System (classic exact-Koopman-embedding example, Brunton et al. 2016 style,
extended here with a control input entering the x1 equation):

    x1_dot = mu * x1 + u
    x2_dot = lambda * (x2 - x1^2)

Claim (standard result, verify by direct differentiation -- see
`lifted_dynamics_matrices` below): the observable vector

    z = [x1, x2, x1^2]

evolves EXACTLY according to a Koopman BILINEAR realization (linear in z,
plus one u*z cross term -- consistent with the `bilinear` flag already
present in models/autoencoder.py::KoopmanAutoencoder):

    z1_dot = mu*z1 + u
    z2_dot = lambda*z2 - lambda*z3
    z3_dot = 2*mu*z3 + 2*z1*u          <-- bilinear cross term: 2*u*z1

i.e. z_dot = A z + B u + u * (C z), with A, B, C given exactly (no fitting,
no approximation) by `lifted_dynamics_matrices`. This is what makes it
useful as a sanity-check limit: PE-transfer here can be checked with
epsilon_phys EXACTLY zero, matching the hypothesis of Shang et al.'s
theorem, before experiments/pe_transfer_sanity_check.py perturbs the lifting
to see how gracefully the rank/PE-transfer margin degrades as
epsilon_phys > 0 is introduced (our framework's actual claim).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def dynamics(t: float, x: np.ndarray, u: float, mu: float = -0.05, lam: float = -1.0) -> np.ndarray:
    x1, x2 = x
    return np.array([mu * x1 + u, lam * (x2 - x1 ** 2)])


def exact_lift(x: np.ndarray) -> np.ndarray:
    """z = [x1, x2, x1^2] -- the exact finite-dimensional Koopman observable
    vector for this system (see module docstring)."""
    x1, x2 = x
    return np.array([x1, x2, x1 ** 2])


def lifted_dynamics_matrices(mu: float = -0.05, lam: float = -1.0) -> dict:
    """Return the EXACT A, B, C matrices such that z_dot = A z + B u + u*(C z),
    for z = exact_lift(x). Derived by direct differentiation (see module
    docstring) -- no fitting involved, this is ground truth for the
    epsilon_phys = 0 sanity-check limit.
    """
    A = np.array([
        [mu, 0.0, 0.0],
        [0.0, lam, -lam],
        [0.0, 0.0, 2 * mu],
    ])
    B = np.array([1.0, 0.0, 0.0])
    C = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    return {"A": A, "B": B, "C": C}


def simulate(x0: np.ndarray, u_fn, t_span: tuple[float, float], dt: float,
             mu: float = -0.05, lam: float = -1.0) -> dict:
    t_eval = np.arange(t_span[0], t_span[1], dt)

    def rhs(t, x):
        return dynamics(t, x, u_fn(t), mu=mu, lam=lam)

    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, rtol=1e-9, atol=1e-11)
    u_vals = np.array([u_fn(t) for t in t_eval])
    return {"t": sol.t, "x": sol.y.T, "u": u_vals}


def generate_pe_trajectory(x0: np.ndarray, t_span: tuple[float, float], dt: float,
                            mu: float = -0.05, lam: float = -1.0, seed: int = 0,
                            excitation: str = "prbs") -> dict:
    """Generate a trajectory under a persistently-exciting input, for the
    Hankel-matrix rank checks in experiments/pe_transfer_sanity_check.py.

    TODO (alongside sims/*.py's other generate_pe_trajectory stubs, Week 1-2):
    implement PRBS/multisine excitation -- shared implementation should
    probably be factored into a sims/_excitation.py helper once 2+ of these
    stubs are filled in, rather than duplicated per-file.
    """
    raise NotImplementedError("Implement alongside other sims/*.py PE-trajectory generators")

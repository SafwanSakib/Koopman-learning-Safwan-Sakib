"""Linear Data-CBF constraint expressed directly on Hankel-matrix rows.

Track B Implementation Plan Sec.2.3: "This is the empirical counterpart to
Theorem 2." Theoretical_Background_Full.md Sec.6.3-6.4.

--- UPDATED 2026-09-10: matches the frozen Track A theory note (theorem2_
data_direct_cbf.tex, eq. 2.15 "the safety-filtered physics-informed DeePC
problem"). Two things changed from the earlier draft of this module: ---

(1) ASSUMPTION A5, RESOLVED (was an open design question, now closed):
    The CBF-type inequality

        e_j^T z_hat_{k+i+1} >= (1 - gamma) * h(x_{k+i}) + rho

    is imposed as a HARD constraint ONLY at i=0 (i.e. only on the very
    first predicted step, e_j^T z_hat_{k+1} >= (1-gamma) h(x_k) + rho).
    The analogous condition for i=1,...,N_h-1 is imposed as a SOFT
    (performance-only, e.g. penalized-slack or simply omitted) term -- it
    plays NO role in Theorem 2's safety certificate. This is proven
    correct, not a simplification: theorem2_data_direct_cbf.tex Remark
    "R1 -- Horizon unrolling is bounded" shows that hard-constraining only
    the first step is exactly what keeps the safety margin eta_1/gamma
    UNIFORM IN k and INDEPENDENT OF THE HORIZON N_h; hard-constraining the
    full horizon would instead require the lifted dynamics to be
    Schur-stable (||A|| < 1), which is not assumed. NO terminal safe set
    is required for this certificate (see controllers/deepc.py's
    DeePCConfig.use_terminal_set, now resolved False by default).

(2) SUPPORT FOR MULTIPLE BARRIER COORDINATES (resolves the min()-combination
    question left open in sims/cartpole.py, sims/cstr.py, sims/quadrotor.py):
    the theory requires h(x) = e_j^T Phi(x) to be a SINGLE LINEAR
    coordinate (notation_assumptions.tex, Sec.Lifting map) -- a hard min()
    over several bounds (e.g. cart-pole's angle bound AND position bound)
    is NOT representable as one such coordinate. The correct resolution,
    consistent with the theory rather than a workaround: embed EACH bound
    as ITS OWN coordinate of Phi (h_coordinates: list[int], see
    models/autoencoder.py), and impose Theorem 2's hard first-step
    constraint INDEPENDENTLY for each coordinate. Theorem 2's proof does
    not depend on which h is chosen, so it applies verbatim,
    coordinate-by-coordinate -- the resulting guarantee is
    h_j(x_k) >= -eta_1/gamma for EVERY embedded barrier j simultaneously,
    which is exactly the AND-of-bounds semantics a min()-combination was
    trying to achieve, without needing a nonlinear (non-embeddable) min.

Key idea (unchanged): because each h_j is one coordinate of the lifted
state z, the corresponding row of the Hankel matrix H_L(z^d) gives
h_hat_j(x_{k+i}) as a LINEAR function of the DeePC decision variable g, at
every step of the horizon, with NO explicit A/B ever formed -- this is the
structural distinction from Folkestad et al. (2020) / Zinage & Bakolas
(2023), who impose the CBF condition on an explicitly identified operator
instead (see Introduction_Literature_Review_Draft.md Sec.1.2.5).

TODO (Week 4, alongside controllers/deepc.py's Koopman-lifted extension):
    - extract_h_rows(hankel_z, h_coordinate, L) -> indices/slice into the
      stacked Hankel matrix corresponding to one h(x) coordinate at each
      horizon step
    - build_cbf_constraint(hankel_z, h_coordinates, gamma, rho, N_h) -> a
      callable compatible with controllers.deepc.DeePCProblem's
      cbf_constraint_builder argument; must return CVXPY-native HARD
      constraints at i=0 for every coordinate in h_coordinates, and
      (optionally) SOFT terms for i=1..N_h-1 -- see signature below.
    - Unit test (tests/test_cbf_constraint.py): verify the constraint is
      provably linear in g (symbolic check via CVXPY's `is_dcp`/affine check,
      or a finite-difference check against direct h(x) evaluation), and
      that it is applied to EVERY coordinate in h_coordinates independently
      when len(h_coordinates) > 1.
"""

from __future__ import annotations

import numpy as np


def extract_h_rows(hankel_z: np.ndarray, h_coordinate: int, lift_dim: int,
                    horizon_start: int, horizon_len: int) -> np.ndarray:
    """Return the sub-block of the stacked lifted Hankel matrix corresponding
    to ONE h(x) observable coordinate, across `horizon_len` future steps
    starting at `horizon_start` (i.e. within the "future" block U_f/Z_f of
    controllers/deepc.py's split_hankel). Called once per entry of
    h_coordinates by build_cbf_constraint below.
    """
    raise NotImplementedError("Week 4")


def build_cbf_constraint(hankel_z: np.ndarray, h_coordinates: list[int], lift_dim: int,
                          gamma: float, rho: float, horizon: int,
                          soft_remaining_horizon: bool = True):
    """Return a function `(g) -> list[cvxpy.Constraint]` implementing, for
    EVERY coordinate j in h_coordinates (plural -- see module docstring
    point (2) on multi-barrier support):

        HARD (i=0 only):
            h_hat_row_{j,1} @ g  >=  (1 - gamma) * (h_hat_row_{j,0} @ g) + rho

        SOFT (i=1..horizon-1, only if soft_remaining_horizon=True):
            included as a penalized slack term in the QP objective, not as
            a hard constraint -- per Assumption A5 (module docstring point
            (1)), these play no role in the safety certificate and exist
            only to discourage the predicted trajectory from planning a
            barrier violation later in the horizon it can't recursively
            back out of.

    Parameters
    ----------
    h_coordinates : list[int]
        Indices into the lifted state z for EACH embedded barrier
        coordinate (models/autoencoder.py's KoopmanAutoencoder.h_coordinates).
        Single-bound benchmarks pass a length-1 list; multi-bound
        benchmarks (cart-pole: angle + position; CSTR: temperature +
        concentration; quadrotor: one per active obstacle) pass one entry
        per bound.
    gamma : float
        Class-K decay rate, 0 < gamma <= 1 (Theoretical_Background_Full.md
        Sec.6.3; Agrawal & Sreenath 2017 for the discrete-time form).
    rho : float
        Tightening parameter (theorem2_data_direct_cbf.tex, Corollary
        "Exact safety via constraint tightening"). rho=0 gives the
        margin-based guarantee h(x_k) >= -eta_1/gamma (Theorem 2); rho set
        to (an estimate of) eta_1 gives exact forward invariance
        h(x_k) >= 0, trading conservatism for an exact guarantee. Both are
        legitimate experimental configurations to report (Research Plan
        Sec.7.4's theory-vs-practice plot).

    Suitable for passing as `cbf_constraint_builder` to
    controllers.deepc.DeePCProblem.
    """
    raise NotImplementedError("Week 4")


def verify_forward_invariance_empirically(
    closed_loop_h_trajectory: np.ndarray, tol: float = 0.0
) -> bool:
    """Post-hoc check used across many random initial conditions / disturbance
    realizations (Track B Sec.2.2's known-model pendulum test, and later the
    full pipeline's safety-violation logging, Sec.3). For multi-barrier
    benchmarks, call this once per barrier coordinate's trajectory (h_j),
    not on a combined/min'd trajectory -- consistent with module docstring
    point (2).
    """
    return bool(np.all(closed_loop_h_trajectory >= -tol))

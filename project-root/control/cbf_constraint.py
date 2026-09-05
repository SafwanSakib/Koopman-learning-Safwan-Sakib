"""Linear Data-CBF constraint expressed directly on Hankel-matrix rows.

Track B Implementation Plan §2.3: "This is the empirical counterpart to
Theorem 2." Theoretical_Background_Full.md §6.3-6.4.

Key idea: because h(x) is embedded as one coordinate of the lifted state z
(models/autoencoder.py's h_coordinate contract), the row of the Hankel
matrix H_L(z^d) corresponding to that coordinate gives h_hat(x_{k+i}) as a
LINEAR function of the DeePC decision variable g, for every step i in the
prediction horizon. So the discrete-time CBF condition

    h(x_{k+i+1}) - h(x_k+i) >= -gamma * h(x_{k+i})     (Theoretical_Background_Full.md §6.3)

becomes a linear inequality constraint on g, at every step of the horizon,
with NO explicit A/B ever formed — this is the structural distinction from
Folkestad et al. (2020) / Zinage & Bakolas (2023), who impose the CBF
condition on an explicitly identified operator instead (see
Introduction_Literature_Review_Draft.md §1.2.5).

TODO (Week 4, alongside control/deepc.py's Koopman-lifted extension):
    - extract_h_rows(hankel_z, h_coordinate, L) -> indices/slice into the
      stacked Hankel matrix corresponding to the h(x) coordinate at each
      horizon step
    - build_cbf_constraint(hankel_z, h_coordinate, gamma, N) -> a callable
      compatible with control.deepc.DeePCProblem's cbf_constraint_builder
      argument; must return CVXPY-native constraint expressions (linear in g)
    - Unit test (tests/test_cbf_constraint.py): verify the constraint is
      provably linear in g (symbolic check via CVXPY's `is_dcp`/affine check,
      or a finite-difference check against direct h(x) evaluation)
"""

from __future__ import annotations

import numpy as np


def extract_h_rows(hankel_z: np.ndarray, h_coordinate: int, lift_dim: int,
                    horizon_start: int, horizon_len: int) -> np.ndarray:
    """Return the sub-block of the stacked lifted Hankel matrix corresponding
    to the h(x) observable coordinate, across `horizon_len` future steps
    starting at `horizon_start` (i.e. within the "future" block U_f/Z_f of
    control/deepc.py's split_hankel).
    """
    raise NotImplementedError("Week 4")


def build_cbf_constraint(hankel_z: np.ndarray, h_coordinate: int, lift_dim: int,
                          gamma: float, horizon: int):
    """Return a function `(g) -> list[cvxpy.Constraint]` implementing

        h_hat_row_{i+1} @ g >= (1 - gamma) * (h_hat_row_i @ g)   for i = 0..horizon-1

    across the prediction horizon, suitable for passing as
    `cbf_constraint_builder` to control.deepc.DeePCProblem.

    Parameters
    ----------
    gamma : float
        Class-K decay rate, 0 < gamma <= 1 (Theoretical_Background_Full.md §6.3).
    """
    raise NotImplementedError("Week 4")


def verify_forward_invariance_empirically(
    closed_loop_h_trajectory: np.ndarray, tol: float = 0.0
) -> bool:
    """Post-hoc check used across many random initial conditions / disturbance
    realizations (Track B §2.2's known-model pendulum test, and later the
    full pipeline's safety-violation logging, §3)."""
    return bool(np.all(closed_loop_h_trajectory >= -tol))

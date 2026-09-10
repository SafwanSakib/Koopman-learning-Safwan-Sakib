"""Core Data-Enabled Predictive Control (DeePC) QP.

Track B Implementation Plan §1.2 (plain linear DeePC on toy LTI system,
Week 1-2 milestone) and §2.3 (Koopman-lifted + Data-CBF version).
Theoretical_Background_Full.md §3.2-3.3.

Two build order per the plan:
    1. Plain linear DeePC on a raw (u, y) trajectory (mass-spring-damper
       toy system) — validates Hankel-matrix construction, u_ini/y_ini
       handling, regularization, against a classical LQR/pole-placement
       controller. NO Koopman, NO CBF at this stage (§1.2).
    2. Koopman-lifted DeePC: same QP machinery but run on (u, z) where
       z = phi_theta(x) (control/deepc.py + models/autoencoder.py), with an
       optional linear Data-CBF constraint injected via
       control/cbf_constraint.py (§2.3, the actual novel object empirically
       validating Theorem 2).

The QP (regularized / "robust" DeePC, Theoretical_Background_Full.md §3.3):

    min_{g, u_f, y_f}  sum_i ell(y_f_i, u_f_i)
                       + lambda_g ||g||_2^2
                       + lambda_sigma ||sigma_y||_2^2
    s.t.  U_p g = u_ini
          Y_p g = y_ini + sigma_y
          U_f g = u_f
          Y_f g = y_f
          u_f in U, y_f in Y
          [optional: linear Data-CBF constraint from cbf_constraint.py]

TODO (Week 1-2 for plain LTI version; Week 4-6 for Koopman+CBF version):
    - build_hankel_matrix(data, L) -> np.ndarray
    - split_hankel(H, T_ini, N) -> (past_block, future_block)
    - DeePCProblem class wrapping the CVXPY problem, with an optional
      `cbf_constraint` argument (a callable from cbf_constraint.py) so the
      safety layer is a strict superset of the plain QP, not a fork of it
      (needed for the "ablation: full method without CBF layer" baseline,
      Research Plan §7.3 item 8 — toggle, don't duplicate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def build_hankel_matrix(data: np.ndarray, depth: int) -> np.ndarray:
    """Build a depth-L block Hankel matrix from a single trajectory.

    Parameters
    ----------
    data : np.ndarray, shape (T, d)
        A length-T trajectory of d-dimensional samples (e.g. inputs u^d or
        lifted states z^d).
    depth : int
        Window depth L (Theoretical_Background_Full.md §2.2).

    Returns
    -------
    np.ndarray, shape (L*d, T-L+1)
    """
    raise NotImplementedError("Week 1-2: implement Hankel matrix construction")


def check_persistency_of_excitation(hankel: np.ndarray, tol: float = 1e-8) -> bool:
    """Full-row-rank check (Theoretical_Background_Full.md §2.3). Used both
    as a data-generation sanity check and as Theorem 1's fallback empirical
    verification procedure (Research Plan §5.4)."""
    raise NotImplementedError("Week 1-2")


@dataclass
class DeePCConfig:
    T_ini: int
    N: int  # prediction horizon
    lambda_g: float = 1.0
    lambda_sigma: float = 1e4
    use_cbf: bool = False  # toggled off for ablation baseline (Research Plan §7.3.8)


class DeePCProblem:
    """Wraps the (regularized) DeePC QP as a CVXPY problem.

    Parameters
    ----------
    u_data, y_data : np.ndarray
        Recorded input/output (or input/lifted-state, for the Koopman
        version) trajectories used to build Hankel matrices.
    config : DeePCConfig
    cbf_constraint_builder : callable | None
        If provided (control/cbf_constraint.py), adds the linear Data-CBF
        constraint to the QP. Leave None to reproduce the "without CBF
        layer" ablation.
    """

    def __init__(self, u_data: np.ndarray, y_data: np.ndarray, config: DeePCConfig,
                 cbf_constraint_builder=None) -> None:
        self.u_data = u_data
        self.y_data = y_data
        self.config = config
        self.cbf_constraint_builder = cbf_constraint_builder
        self._problem = None  # cvxpy.Problem, built lazily

    def build(self) -> None:
        raise NotImplementedError("Week 1-2 (plain), Week 4-6 (+ CBF constraint)")

    def solve(self, u_ini: np.ndarray, y_ini: np.ndarray) -> dict:
        """Solve the QP given the most recent u_ini/y_ini window; return the
        optimal first input plus diagnostics (solve time, feasibility,
        predicted h(x) trajectory if CBF is active)."""
        raise NotImplementedError("Week 1-2 (plain), Week 4-6 (+ CBF constraint)")

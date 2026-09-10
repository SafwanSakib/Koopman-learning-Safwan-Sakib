"""Core Data-Enabled Predictive Control (DeePC) QP.

Track B Implementation Plan Sec.1.2 (plain linear DeePC on toy LTI system,
Week 1-2 milestone) and Sec.2.3 (Koopman-lifted + Data-CBF version).
Theoretical_Background_Full.md Sec.3.2-3.3.

--- UPDATED 2026-09-10: frozen Track A theory note received (theory_note.tex
+ notation_assumptions.tex + theorem1/2/3 + R5 resolution memo). This module
now matches the frozen notation and two REQUIRED (not optional) modelling
decisions from that note -- see inline citations below. ---

Two build order per the plan:
    1. Plain linear DeePC on a raw (u, y) trajectory (mass-spring-damper
       toy system) -- validates Hankel-matrix construction, u_ini/y_ini
       handling, regularization, against a classical LQR/pole-placement
       controller. NO Koopman, NO CBF at this stage (Sec.1.2).
    2. Koopman-lifted DeePC: same QP machinery but run on (u, z) where
       z = phi_theta(x) (controllers/deepc.py + models/autoencoder.py), with an
       optional linear Data-CBF constraint injected via
       controllers/cbf_constraint.py (Sec.2.3, the actual novel object
       empirically validating Theorem 2).

Frozen notation (notation_assumptions.tex): N = lifting dimension
(models/autoencoder.py's lift_dim), so the prediction horizon is N_h here
-- NOT N -- to avoid the symbol clash. T_ini is the initial-condition
window length; L = T_ini + N_h is the total Hankel depth.

The QP (regularized / "robust" DeePC, theorem2_data_direct_cbf.tex eq. 2.1,
matching Theoretical_Background_Full.md Sec.3.3's structure):

    min_{g, sigma_z}  sum_i ||z_hat_{k+i} - z_ref||_Q^2 + sum_i ||u_hat_{k+i}||_R^2
                       + lambda_sigma ||sigma_z||^2 + lambda_g ||g||^2
    s.t.  U_p g = u_ini
          Z_p g = z_ini + sigma_z
          U_f g = u_f  in  U^{N_h}
          Z_f g = z_f
          [Data-CBF constraint, hard at i=1 only -- see cbf_constraint.py]

TWO REQUIRED MODELLING DECISIONS (not implementation details -- both are
load-bearing for Theorem 2/3's guarantees to hold; do not silently drop
either while implementing):

    (P1) Data-normalized regularization (notation_assumptions.tex
    Sec.P1, "Prescription P1"): lambda_g = lambda_g0 * l and
    lambda_sigma = lambda_sigma0 * l, where l = T - L + 1 is the number of
    Hankel columns (equivalently: use an AVERAGED objective
    (1/l)*tracking_cost + lambda_g0*||g||^2). This is a PREREQUISITE for
    the safety margin (Theorem 3) to be well-behaved with T at all
    (theorem3_step0_R5_resolution.tex, Lemma "Coefficient scaling" --
    with a FIXED lambda_g the bound does not decay). DeePCConfig below
    takes lambda_g0/lambda_sigma0, not raw lambda_g/lambda_sigma, and
    DeePCProblem.build() must compute lambda_g = lambda_g0 * l internally
    once l is known from the data.

    (A5 -- resolved, no longer pending) The CBF-type constraint from
    controllers/cbf_constraint.py is HARD only on the first predicted step
    (i=1); the corresponding condition for i=2..N_h is imposed as a SOFT
    (performance) term only. No terminal safe set is required for the
    safety certificate itself (Theorem 2 is proven without one) -- a
    terminal set is only relevant to an OPTIONAL full-horizon QP
    recursive-feasibility robustness property (Bajelani & van Heusden
    2023's warm-start-tail argument), which is a stretch goal, not part
    of the base method. See controllers/cbf_constraint.py for the constraint
    construction itself.

Also load-bearing but a DIAGNOSTIC rather than a QP-construction change:

    (A6 -- controllability of the lifted pair (A,B)) required for Theorem
    1's PE-transfer result to apply (via Shang, Cortes, Zheng 2024).
    check_controllability() below is a diagnostic to run after training
    models/autoencoder.py's A,B (Week 2-4) -- not used inside the QP.

    (A7 -- normalized excitation) strengthens "full rank" to "full rank
    at a rate": sigma_min(H*) >= mu0 * sqrt(l) for some T-independent
    mu0>0. check_normalized_excitation() below estimates this rate
    empirically from a candidate dataset; it is the practical, checkable
    stand-in for A7 needed before trusting Theorem 3's margin bound on
    real (finite-T) data.

OPEN ITEM (theorem3_step0_R5_resolution.tex, Remark "A well-posed open
problem, not a hole"): the O(1/sqrt(T)) noise-decay rate for the ONLINE
measurement-noise term is only proven under a DECOUPLING hypothesis: g is
computed from data independent of the noise realization entering the
online prediction (e.g. g from one half of the trajectory, prediction
evaluated using the other half). Without decoupling, Theorem 3 still holds
but the bound is T-independent past a threshold (still a valid, if less
sharp, guarantee). `DeePCConfig.decouple_g_data` below is a flag for this;
see experiments/theorem3_margin_validation.py (added 2026-09-10) for the
experiment comparing the two regimes against the theoretical prediction.

TODO (Week 1-2 for plain LTI version; Week 4-6 for Koopman+CBF version):
    - build_hankel_matrix(data, L) -> np.ndarray
    - split_hankel(H, T_ini, N_h) -> (past_block, future_block)
    - DeePCProblem class wrapping the CVXPY problem, with an optional
      `cbf_constraint` argument (a callable from cbf_constraint.py) so the
      safety layer is a strict superset of the plain QP, not a fork of it
      (needed for the "ablation: full method without CBF layer" baseline,
      Research Plan Sec.7.3 item 8 -- toggle, don't duplicate).
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
        Window depth L (Theoretical_Background_Full.md Sec.2.2).

    Returns
    -------
    np.ndarray, shape (L*d, T-L+1)
    """
    raise NotImplementedError("Week 1-2: implement Hankel matrix construction")


def check_persistency_of_excitation(hankel: np.ndarray, tol: float = 1e-8) -> bool:
    """Full-row-rank check (Theoretical_Background_Full.md Sec.2.3). Used both
    as a data-generation sanity check and as Theorem 1's fallback empirical
    verification procedure (Research Plan Sec.5.4)."""
    raise NotImplementedError("Week 1-2")


def check_normalized_excitation(hankel_star: np.ndarray, num_columns: int) -> dict:
    """Assumption A7 diagnostic (notation_assumptions.tex Sec.A7): estimate
    whether sigma_min(hankel_star) grows at rate ~ mu0 * sqrt(l), l =
    num_columns, for a T-independent mu0 > 0.

    This is meant to be called across a SWEEP of T (i.e. re-run with growing
    prefixes of a long trajectory) so the caller can regress
    sigma_min / sqrt(l) against l and check it floors at a positive constant
    rather than decaying -- a single call only returns the raw ratio for
    one T; see experiments/pe_transfer_sanity_check.py and
    experiments/theorem3_margin_validation.py for the sweep-based callers.

    Returns
    -------
    dict with keys: "sigma_min", "l", "sigma_min_over_sqrt_l".
    """
    raise NotImplementedError(
        "Week 1-2, alongside check_persistency_of_excitation: compute "
        "np.linalg.svd(hankel_star)[1].min() and the normalized ratio."
    )


def check_controllability(A: np.ndarray, B: np.ndarray, tol: float = 1e-8) -> dict:
    """Assumption A6 diagnostic (notation_assumptions.tex Sec.A6): check the
    learned lifted pair (A, B) is controllable, via the controllability
    matrix [B, AB, A^2 B, ..., A^{N-1} B] having full row rank N.

    Required for Theorem 1 (PE-transfer) to apply via Shang, Cortes, Zheng
    (2024) -- run this as a diagnostic after every models/autoencoder.py
    training run (Week 2-4), not just once; a trained (A,B) is not
    guaranteed controllable just because the architecture allows it.

    Uses the `control` package (already in environment.yml) for the
    controllability matrix, consistent with the rest of the stack rather
    than hand-rolling it.

    Returns
    -------
    dict with keys: "controllable" (bool), "rank", "full_rank" (= N).
    """
    import control as ct  # local import: only needed by this diagnostic

    N = A.shape[0]
    ctrb_matrix = ct.ctrb(A, B)
    rank = np.linalg.matrix_rank(ctrb_matrix, tol=tol)
    return {"controllable": rank == N, "rank": rank, "full_rank": N}


@dataclass
class DeePCConfig:
    T_ini: int
    N_h: int  # prediction horizon (renamed from N -- see module docstring,
              # avoids clash with models.autoencoder.KoopmanAutoencoder's
              # lift_dim N, per notation_assumptions.tex's fixed notation)
    lambda_g0: float = 1.0
    lambda_sigma0: float = 1e4
    # PRESCRIPTION P1 (module docstring, REQUIRED): actual QP weights are
    # lambda_g0 * l, lambda_sigma0 * l with l = T - L + 1 computed at
    # DeePCProblem.build() time from the data actually supplied -- these
    # two fields are NOT the raw QP weights, do not use them as such.
    use_cbf: bool = False  # toggled off for ablation baseline (Research Plan Sec.7.3.8)
    gamma: float = 0.5  # discrete-CBF rate (Agrawal & Sreenath 2017), 0 < gamma <= 1
    rho: float = 0.0  # CBF tightening parameter (theorem2_data_direct_cbf.tex
                       # Corollary "Exact safety via constraint tightening"):
                       # rho=0 -> margin-based safety (Theorem 2, h(x_k) >= -m);
                       # rho=eta_1 (computed externally, e.g. from Theorem 3's
                       # bound) -> exact forward invariance, h(x_k) >= 0 always,
                       # at the cost of extra conservatism. Both are legitimate
                       # experimental configurations -- Research Plan Sec.7.4's
                       # "theory-vs-practice" plot should report both.
    use_terminal_set: bool = False
    # RESOLVED 2026-09-10 (was "pending" -- see Assumption A5 in the module
    # docstring): the base safety certificate (Theorem 2) needs NO terminal
    # set. Leave this False for the main method and all safety-relevant
    # experiments. Only set True for the OPTIONAL Bajelani & van Heusden
    # (2023)-style full-horizon recursive-feasibility stretch goal, which is
    # a robustness nicety, not required for any of the paper's theorems.
    decouple_g_data: bool = False
    # OPEN ITEM (module docstring): if True, split the offline trajectory in
    # half, solve for g using only the first half's Hankel columns, and
    # evaluate/report the online prediction error using held-out data drawn
    # from the second half only. This is what Proposition
    # "Noise contribution under decoupling" (theorem3_step0_R5_resolution.tex)
    # requires for the tighter O(sigma/sqrt(T)) margin term to be provably
    # valid; the un-decoupled default still gives a valid (Theorem-3-backed)
    # margin, just without that specific rate guarantee. See
    # experiments/theorem3_margin_validation.py for the comparison experiment.


class DeePCProblem:
    """Wraps the (regularized) DeePC QP as a CVXPY problem.

    Parameters
    ----------
    u_data, y_data : np.ndarray
        Recorded input/output (or input/lifted-state, for the Koopman
        version) trajectories used to build Hankel matrices.
    config : DeePCConfig
    cbf_constraint_builder : callable | None
        If provided (controllers/cbf_constraint.py), adds the linear Data-CBF
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
        """Build the CVXPY problem.

        MUST implement Prescription P1 here: compute l = T - L + 1 from the
        actual data length and depth, then set the QP's regularization
        weights to config.lambda_g0 * l and config.lambda_sigma0 * l (or
        equivalently use the averaged-objective form -- see module
        docstring) -- do NOT use config.lambda_g0/lambda_sigma0 directly as
        the QP weights.

        If config.decouple_g_data is True, this must also split u_data/
        y_data into two halves and build separate Hankel matrices for the
        "compute g" role vs. the "evaluate prediction error" role (see
        DeePCConfig.decouple_g_data docstring).
        """
        raise NotImplementedError("Week 1-2 (plain), Week 4-6 (+ CBF constraint)")

    def solve(self, u_ini: np.ndarray, y_ini: np.ndarray) -> dict:
        """Solve the QP given the most recent u_ini/y_ini window; return the
        optimal first input plus diagnostics (solve time, feasibility,
        predicted h(x) trajectory if CBF is active)."""
        raise NotImplementedError("Week 1-2 (plain), Week 4-6 (+ CBF constraint)")

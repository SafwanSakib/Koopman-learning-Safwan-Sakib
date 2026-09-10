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
    data : np.ndarray, shape (T, d) or (T,)
        A length-T trajectory of d-dimensional samples (e.g. inputs u^d or
        lifted states z^d). A 1-D array is treated as d=1.
    depth : int
        Window depth L (Theoretical_Background_Full.md Sec.2.2).

    Returns
    -------
    np.ndarray, shape (L*d, T-L+1). Column i is the flattened length-L
    window data[i:i+L] (row-major: block for time i, then i+1, ..., i+L-1).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    T, d = data.shape
    L = depth
    if T < L:
        raise ValueError(f"trajectory length T={T} is shorter than depth L={L}")
    num_cols = T - L + 1
    H = np.empty((L * d, num_cols))
    for i in range(num_cols):
        H[:, i] = data[i:i + L].reshape(-1)
    return H


def split_hankel(hankel: np.ndarray, block_dim: int, T_ini: int, N_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a depth-(T_ini+N_h) block Hankel matrix into its past
    (first T_ini blocks of rows) and future (remaining N_h blocks) parts,
    e.g. splitting H_L(u^d) into U_p, U_f (Theoretical_Background_Full.md
    Sec.3.1). `block_dim` is the per-timestep dimension d used when the
    Hankel matrix was built (build_hankel_matrix's `data.shape[1]`).
    """
    past_rows = T_ini * block_dim
    return hankel[:past_rows, :], hankel[past_rows:, :]


def check_persistency_of_excitation(hankel: np.ndarray, tol: float = 1e-8) -> bool:
    """Full-row-rank check (Theoretical_Background_Full.md Sec.2.3). Used both
    as a data-generation sanity check and as Theorem 1's fallback empirical
    verification procedure (Research Plan Sec.5.4)."""
    rank = np.linalg.matrix_rank(hankel, tol=tol)
    return bool(rank == hankel.shape[0])


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
    singular_values = np.linalg.svd(hankel_star, compute_uv=False)
    sigma_min = float(singular_values.min())
    l = num_columns
    return {
        "sigma_min": sigma_min,
        "l": l,
        "sigma_min_over_sqrt_l": sigma_min / np.sqrt(l),
    }


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

        Implements Prescription P1: l = T - L + 1 is computed from the
        actual data supplied, and the QP's regularization weights are set
        to config.lambda_g0 * l and config.lambda_sigma0 * l (NOT
        config.lambda_g0/lambda_sigma0 directly).

        If config.decouple_g_data is True, the offline trajectory is split
        in half: g is solved for using only the FIRST half's Hankel
        columns; the second half is stashed on self.eval_u_data /
        self.eval_y_data for external (caller-side) held-out evaluation
        (see experiments/theorem3_margin_validation.py) -- this class does
        not itself evaluate held-out prediction error, it only performs the
        data split.

        If config.use_cbf is True and a cbf_constraint_builder was supplied
        (controllers/cbf_constraint.py), its returned constraints are added
        to the QP verbatim -- the safety layer is a strict superset of the
        plain QP built here, never a separate code path (Research Plan
        Sec.7.3 item 8, "ablation: full method without CBF layer" is simply
        use_cbf=False on this same class).
        """
        import cvxpy as cp

        cfg = self.config
        u_data = np.asarray(self.u_data, dtype=float)
        y_data = np.asarray(self.y_data, dtype=float)
        if u_data.ndim == 1:
            u_data = u_data.reshape(-1, 1)
        if y_data.ndim == 1:
            y_data = y_data.reshape(-1, 1)

        if cfg.decouple_g_data:
            T_total = u_data.shape[0]
            half = T_total // 2
            g_u_data, g_y_data = u_data[:half], y_data[:half]
            self.eval_u_data, self.eval_y_data = u_data[half:], y_data[half:]
        else:
            g_u_data, g_y_data = u_data, y_data

        m = g_u_data.shape[1]
        p = g_y_data.shape[1]
        L = cfg.T_ini + cfg.N_h

        Hu = build_hankel_matrix(g_u_data, L)
        Hy = build_hankel_matrix(g_y_data, L)
        l = Hu.shape[1]  # number of Hankel columns, per Prescription P1

        Up, Uf = split_hankel(Hu, m, cfg.T_ini, cfg.N_h)
        Yp, Yf = split_hankel(Hy, p, cfg.T_ini, cfg.N_h)

        self.m, self.p, self.l = m, p, l
        self.Up, self.Uf, self.Yp, self.Yf = Up, Uf, Yp, Yf

        g = cp.Variable(l)
        sigma_y = cp.Variable(cfg.T_ini * p)

        u_ini_param = cp.Parameter(cfg.T_ini * m)
        y_ini_param = cp.Parameter(cfg.T_ini * p)
        y_ref_param = cp.Parameter(cfg.N_h * p, value=np.zeros(cfg.N_h * p))

        y_f = Yf @ g
        u_f = Uf @ g

        # Prescription P1 (REQUIRED, see module docstring): weights scale
        # with l, the number of Hankel columns actually used to solve for g.
        lambda_g = cfg.lambda_g0 * l
        lambda_sigma = cfg.lambda_sigma0 * l

        tracking_cost = cp.sum_squares(y_f - y_ref_param)
        input_cost = cp.sum_squares(u_f)
        reg_cost = lambda_g * cp.sum_squares(g) + lambda_sigma * cp.sum_squares(sigma_y)
        objective = cp.Minimize(tracking_cost + input_cost + reg_cost)

        constraints = [Up @ g == u_ini_param, Yp @ g == y_ini_param + sigma_y]
        if cfg.use_cbf and self.cbf_constraint_builder is not None:
            constraints += self.cbf_constraint_builder(g)

        self._problem = cp.Problem(objective, constraints)
        self._g = g
        self._sigma_y = sigma_y
        self._u_ini_param = u_ini_param
        self._y_ini_param = y_ini_param
        self._y_ref_param = y_ref_param

    def set_reference(self, y_ref: np.ndarray) -> None:
        """Update the tracked output reference (defaults to zero, i.e.
        regulation to the origin, if never called). y_ref must have shape
        (N_h, p) or be flattenable to length N_h * p."""
        if self._problem is None:
            self.build()
        self._y_ref_param.value = np.asarray(y_ref, dtype=float).reshape(-1)

    def solve(self, u_ini: np.ndarray, y_ini: np.ndarray) -> dict:
        """Solve the QP given the most recent u_ini/y_ini window; return the
        optimal first input plus diagnostics (solve time, feasibility,
        predicted h(x) trajectory if CBF is active). Delegates the actual
        solver call (with fallback) to controllers.qp_solver.solve_qp, so
        solver choice/fallback logic lives in one place (see that module's
        docstring)."""
        from controllers.qp_solver import solve_qp

        if self._problem is None:
            self.build()

        self._u_ini_param.value = np.asarray(u_ini, dtype=float).reshape(-1)
        self._y_ini_param.value = np.asarray(y_ini, dtype=float).reshape(-1)

        solve_info = solve_qp(self._problem)

        g_val = self._g.value
        if g_val is None:
            return {**solve_info, "u_first": None}

        u_f_val = self.Uf @ g_val
        y_f_val = self.Yf @ g_val
        return {
            **solve_info,
            "u_first": u_f_val[: self.m],
            "u_f": u_f_val,
            "y_f": y_f_val,
            "g": g_val,
            "sigma_y": self._sigma_y.value,
        }

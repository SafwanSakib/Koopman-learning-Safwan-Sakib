"""Theorem 3 margin validation experiment: decoupled vs. standard data usage.

Added 2026-09-10, per the frozen Track A theory note
(theorem3_step0_R5_resolution.tex, Remark "A well-posed open problem, not
a hole"), confirmed by user 2026-09-10: "Yes, add it as a new experiment
module."

What the open problem is (full detail in controllers/deepc.py's module
docstring and DeePCConfig.decouple_g_data): Theorem 3's tightest form of
the online noise term, O(sigma / sqrt(T)), is only PROVEN under a
DECOUPLING hypothesis -- the coefficient vector g must be computed from
data statistically independent of the noise realization entering the
online prediction it's used to make. Without decoupling, Theorem 3 still
gives a valid margin bound, but it is T-INDEPENDENT past a computable
threshold (the bound floors rather than continuing to shrink with more
data) -- see theorem3_margin_bound.tex, "Data volume is the weakest of the
three levers."

This experiment makes that distinction EMPIRICAL rather than purely
theoretical:

    1. STANDARD (coupled): solve the DeePC QP the usual way -- g computed
       from the full trajectory's Hankel matrix, same data used both to
       build g and to realize the online prediction. Sweep trajectory
       length T, measure the ACHIEVED safety margin (min observed h(x_k)
       over many closed-loop rollouts, per benchmark, per T), and compare
       against the theoretical bound WITHOUT the decoupling refinement
       (theorem3_margin_bound.tex's unconditional statement of Theorem 3).

    2. DECOUPLED: split each trajectory in half; compute g from the first
       half's Hankel columns only (DeePCConfig.decouple_g_data=True in
       controllers/deepc.py); evaluate the online prediction / closed-loop
       rollout using only held-out data drawn from the second half. Sweep
       T (now meaning "T per half," to keep total data collected
       comparable to the standard case), measure the achieved margin, and
       compare against the SHARPER theoretical bound (the
       O(sigma/sqrt(T)) conditional refinement, theorem3_margin_bound.tex
       eq. with C_1/sqrt(T-L+1)).

    3. FIGURE: achieved margin vs. T, TWO curves (standard, decoupled),
       each with its OWN matching theoretical-bound overlay line -- this
       is the direct empirical test of whether decoupling buys the
       predicted improvement in practice, and (separately) whether either
       theoretical bound is a reasonably tight description of what's
       actually achieved. This feeds the same "theory-vs-practice" slot in
       Research Plan Sec.7.4 as the Theorem 1 sanity check
       (experiments/pe_transfer_sanity_check.py) does for PE-transfer.

Relationship to other modules: depends on controllers/deepc.py's
DeePCProblem (with decouple_g_data toggled) and
controllers/cbf_constraint.py's build_cbf_constraint (for the closed-loop
safety-margin rollouts) both being implemented (Week 4-6), and on
models/losses.py's held_out_residual_bound (Week 2-4, already implemented)
for the C_3 term of whichever theoretical bound is being overlaid. Until
those land, this module documents the exact experimental protocol so
Track A and Track B stay in sync on what "the margin validation" concretely
means -- same pattern as experiments/pe_transfer_sanity_check.py.

Benchmark choice: run this on van_der_pol or cartpole first (whichever's
end-to-end pipeline (Track B Sec.3) is working first) with a SINGLE
embedded barrier coordinate, not a multi-bound benchmark -- keeps the
margin comparison uncluttered by having to track multiple h_j
simultaneously. Extend to multi-bound benchmarks only after the single-
barrier version is validated.

TODO:
    - Implement once controllers/deepc.py's DeePCProblem (incl. decouple_g_data)
      and controllers/cbf_constraint.py's build_cbf_constraint are done
      (Week 4-6)
    - Decide the T-sweep range jointly with experiments/sweep.py's main
      grid (Research Plan Sec.7.2) so this doesn't duplicate compute --
      likely reuse the same closed-loop rollout infrastructure
      (experiments/run_experiment.py) with decouple_g_data as an extra
      sweep dimension rather than a fully separate runner.
    - Coordinate with Track A on the exact numeric values of C_1, C_2,
      C_2', C_3 (theorem3_margin_bound.tex's Theorem "Finite-sample
      noise-to-safety-margin bound") once ||A||, L_phi, lambda_g0, J_circ
      etc. are known for the trained model, so the overlay line uses the
      real constants rather than placeholders.
"""

from __future__ import annotations

import numpy as np


def run_standard_sweep(benchmark: str, T_values: list[int], num_seeds: int = 10,
                        **rollout_kwargs) -> dict:
    """Stage 1 (standard/coupled): sweep T, run closed-loop rollouts with
    the ordinary (non-decoupled) DeePCConfig, measure achieved margin
    (min observed h(x_k)) per T, aggregated across num_seeds seeds.

    Returns a dict keyed by T with {"achieved_margin_mean", "achieved_margin_std"}.
    """
    raise NotImplementedError(
        "Depends on controllers.deepc.DeePCProblem and controllers.cbf_constraint."
        "build_cbf_constraint (Week 4-6). See module docstring, Stage 1."
    )


def run_decoupled_sweep(benchmark: str, T_values: list[int], num_seeds: int = 10,
                         **rollout_kwargs) -> dict:
    """Stage 2 (decoupled): same protocol as run_standard_sweep, but with
    DeePCConfig.decouple_g_data=True -- g computed from the first half of
    each trajectory, closed-loop rollout / margin measurement using only
    the second half's data for online prediction evaluation.
    """
    raise NotImplementedError(
        "Depends on controllers.deepc.DeePCProblem's decouple_g_data support "
        "(Week 4-6). See module docstring, Stage 2."
    )


def compute_theoretical_bound(T: int, decoupled: bool, sigma: float,
                               epsilon_bar: float, gamma: float,
                               model_constants: dict) -> float:
    """Evaluate Theorem 3's bound (theorem3_margin_bound.tex) at a given T,
    either the unconditional form (decoupled=False, T-independent past a
    threshold) or the conditional decoupled form (decoupled=True,
    C_1/sqrt(T-L+1) term included).

    `model_constants` must supply the pieces needed for C_1, C_2, C_2', C_3
    (||A||, L_Phi, lambda_g0, J_circ, T_ini, L, delta) -- see
    theorem3_margin_bound.tex's Theorem statement for the exact formulas.
    Coordinate with Track A before filling these in for real (see module
    docstring TODO).
    """
    raise NotImplementedError("Coordinate with Track A on exact constants; see module docstring")


def make_margin_validation_figure(standard_results: dict, decoupled_results: dict,
                                   theoretical_overlay: dict, out_path: str) -> None:
    """Stage 3: render the T vs. achieved-margin figure, two empirical
    curves (standard, decoupled) each with its matching theoretical-bound
    overlay line.
    """
    raise NotImplementedError("Implement once Stages 1/2 produce real data")

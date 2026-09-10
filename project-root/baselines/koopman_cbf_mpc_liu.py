"""Baseline #10 (NEW, added 2026-09-05) -- Koopman-based linear MPC for safe
control using CBF, receding-horizon, indirect (Liu, Wu, Zhang, Drgona, Belta,
2026, arXiv:2603.21070).

Added per Track C's Weeks 1-2 literature lock
(Consolidated_Findings_TrackA_TrackB_Revisions.md, Part 1 finding #3 and
Part 3.1) -- decision made 2026-09-05: add as a NEW baseline #10, keep the
original Folkestad/Zinage-Bakolas single-step baseline (#3,
baselines/koopman_cbf_indirect.py) as well, since the two isolate different
experimental variables.

Why this baseline matters (do not treat it as a redundant twin of #3):
    - It already combines Koopman lifting + CBF + a receding-horizon
      predictive-control structure (indirect: identifies A, B explicitly,
      then plans over a horizon), unlike Folkestad/Zinage-Bakolas which are
      single-step reactive filters.
    - This means a comparison of OUR method against THIS baseline holds the
      "receding-horizon" variable constant on both sides, isolating
      EXACTLY two things: (a) direct/behavioral (ours) vs. indirect/
      identified-operator (this baseline), and (b) physics-informed (ours)
      vs. not (this baseline lacks a physics-informed term entirely, per
      the literature-lock gap table).
    - It therefore replaces Baseline #3 as the primary "closest indirect
      competitor" in the results narrative and in any figure making the
      direct-vs-indirect argument -- keep #3 around only for the secondary
      "does predictive horizon itself matter" ablation story.

Consolidated_Findings gap-table entry for this paper (for reference when
building Track B's own results table, mirroring paper Table 1):
    direct/behavioral: NO (indirect) | Koopman: YES | receding-horizon: YES
    | CBF: YES | physics-informed: NO

TODO (Week 6-10, alongside baselines/koopman_cbf_indirect.py):
    - Read arXiv:2603.21070 in full before implementing (Research Plan
      Sec.4.4 citation-verification rule applies here as much as to any
      other baseline).
    - Implement: explicit Koopman A,B identification (reuse
      models/autoencoder.py's encoder/A/B without the DeePC layer, same
      pattern as baselines/koopman_mpc_indirect.py), then a receding-horizon
      linear MPC with the CBF condition enforced as a linear constraint
      across the horizon in z-space (structurally similar to
      controllers/cbf_constraint.py's constraint form, but applied to the
      explicit A,B recursion instead of Hankel-matrix rows).
    - Fairness discipline (Track B Sec.4.2): tune horizon length / CBF gamma
      with comparable effort to the main method, since this is now the
      single most important indirect-method comparison in the paper.
"""

from __future__ import annotations


def run(benchmark: str, config: dict) -> dict:
    raise NotImplementedError(
        "Week 6-10: implement Liu et al. 2026 Koopman-CBF-MPC baseline. "
        "See module docstring -- now the primary indirect-method comparison."
    )

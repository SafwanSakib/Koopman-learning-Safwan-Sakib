"""Baseline #3 -- Koopman-CBF (indirect, identified-operator, SINGLE-STEP).

Research Plan Sec.7.3.3 / Track B Sec.4.2 row 3: Folkestad-style /
Zinage-Bakolas-style -- explicit identified A, B first, then a CBF-QP filter
on top. Isolates "direct vs. indirect" as an experimental variable.

Keep the CBF formulation itself (gamma, constraint structure) as close as
possible to the main method's controllers/cbf_constraint.py so the comparison
isolates exactly one variable: whether the constraint is imposed on an
explicit operator (this baseline) or directly on Hankel-matrix rows (ours).

--- UPDATED 2026-09-05 per Track C literature lock (Consolidated_Findings_
TrackA_TrackB_Revisions.md, Sec.3.1) ---
This baseline is now explicitly the SINGLE-STEP reactive member of the
Koopman-CBF family: Folkestad et al. (2020) and Zinage & Bakolas (2023) both
apply the CBF condition once, not across a receding predictive horizon. Liu,
Wu, Zhang, Drgona, Belta (2026, arXiv:2603.21070) subsequently closed that
specific gap with an indirect Koopman-CBF-MPC that DOES have a receding
horizon -- see baselines/koopman_cbf_mpc_liu.py (Baseline #10), added
alongside this one rather than replacing it (decision made 2026-09-05: keep
both, since they isolate different variables -- see that module's docstring).

Consequence for what this baseline demonstrates: the direct-vs-indirect
comparison against OUR method is now most cleanly made against Baseline #10
(receding-horizon held constant on both sides). This baseline (#3) remains
useful as the ablation that additionally isolates the effect of the
predictive-horizon itself (single-step filter vs. multi-step DeePC-based
predictor), which Baseline #10 alone would not show.

TODO (Week 6-10):
    - Reuse controllers/cbf_constraint.py's *mathematical form* but apply it to
      z_{k+1} = A z_k + B u_k directly (explicit operator), not to Hankel
      rows -- this is the "Zinage & Bakolas" style baseline referenced
      throughout Theoretical_Background_Full.md Part 7.2.
"""

from __future__ import annotations


def run(benchmark: str, config: dict) -> dict:
    raise NotImplementedError("Week 6-10: implement indirect Koopman-CBF baseline.")

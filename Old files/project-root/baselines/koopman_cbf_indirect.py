"""Baseline #3 -- Koopman-CBF (indirect, identified-operator).

Research Plan Sec.7.3.3 / Track B Sec.4.2 row 3: Folkestad-style /
Zinage-Bakolas-style -- explicit identified A, B first, then a CBF-QP filter
on top. Isolates "direct vs. indirect" as an experimental variable.

Keep the CBF formulation itself (gamma, constraint structure) as close as
possible to the main method's control/cbf_constraint.py so the comparison
isolates exactly one variable: whether the constraint is imposed on an
explicit operator (this baseline) or directly on Hankel-matrix rows (ours).

TODO (Week 6-10):
    - Reuse control/cbf_constraint.py's *mathematical form* but apply it to
      z_{k+1} = A z_k + B u_k directly (explicit operator), not to Hankel
      rows -- this is the "Zinage & Bakolas" style baseline referenced
      throughout Theoretical_Background_Full.md Part 7.2.
"""

from __future__ import annotations


def run(benchmark: str, config: dict) -> dict:
    raise NotImplementedError("Week 6-10: implement indirect Koopman-CBF baseline.")

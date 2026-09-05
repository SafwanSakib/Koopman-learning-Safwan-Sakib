"""Thin CVXPY/OSQP wrapper shared by control/deepc.py and the CBF-QP filter
(Track B §2.2's known-model pendulum test).

Kept as a separate module so solver choice/backend (OSQP vs. CasADi fallback,
Research Plan §8) and solve-time instrumentation (needed for the
"Computation: solve time per control step" metric, Research Plan §7.4) live
in one place instead of being duplicated across control/deepc.py and every
baseline.

TODO (Week 1-2):
    - solve_qp(problem: cvxpy.Problem, solver="OSQP", **kwargs) -> dict with
      status, solve_time, optimal value, and variable values
    - Fallback logic: if OSQP fails/infeasible, optionally retry with a
      different solver (ECOS, SCS) before reporting infeasibility upstream —
      log which solver actually succeeded, since infeasibility-vs-solver-quirk
      is a common source of spurious "safety violation" results.
"""

from __future__ import annotations

import time


def solve_qp(problem, solver: str = "OSQP", **solver_kwargs) -> dict:
    """Solve a CVXPY problem, returning a dict with at least:
    {status, solve_time_seconds, value, variables}.
    """
    raise NotImplementedError("Week 1-2: implement solver wrapper + timing")


class SolveTimer:
    """Context manager for timing a single QP solve, for the
    per-control-step computation metric (Research Plan §7.4)."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_seconds = time.perf_counter() - self._t0
        return False

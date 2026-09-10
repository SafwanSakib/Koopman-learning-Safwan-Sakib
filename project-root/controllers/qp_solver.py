"""Thin CVXPY/OSQP wrapper shared by controllers/deepc.py and the CBF-QP filter
(Track B Sec.2.2's known-model pendulum test).

Kept as a separate module so solver choice/backend (OSQP vs. CasADi fallback,
Research Plan Sec.8) and solve-time instrumentation (needed for the
"Computation: solve time per control step" metric, Research Plan Sec.7.4) live
in one place instead of being duplicated across controllers/deepc.py and every
baseline.
"""

from __future__ import annotations

import time


def solve_qp(problem, solver: str = "OSQP", fallback_solvers: tuple[str, ...] = ("ECOS", "SCS"),
             **solver_kwargs) -> dict:
    """Solve a CVXPY problem, returning a dict with at least:
    {status, solver_used, solve_time_seconds, value}.

    Tries `solver` first; if it fails to reach an optimal (or
    optimal_inaccurate) status, or raises, falls back through
    `fallback_solvers` in order. Reports which solver actually succeeded --
    per the module docstring, infeasibility-vs-solver-quirk is a common
    source of spurious "safety violation" results, so this is logged
    explicitly rather than silently retried.
    """
    solvers_to_try = [solver] + [s for s in fallback_solvers if s != solver]
    last_error: Exception | None = None

    for candidate_solver in solvers_to_try:
        try:
            with SolveTimer() as timer:
                problem.solve(solver=candidate_solver, **solver_kwargs)
            if problem.status in ("optimal", "optimal_inaccurate"):
                return {
                    "status": problem.status,
                    "solver_used": candidate_solver,
                    "solve_time_seconds": timer.elapsed_seconds,
                    "value": problem.value,
                }
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
            # solver backend can raise for reasons unrelated to the model
            # (missing binary, unsupported cone, etc.); we want to try the
            # next candidate rather than abort the whole control step.
            last_error = exc
            continue

    return {
        "status": getattr(problem, "status", "failed"),
        "solver_used": None,
        "solve_time_seconds": None,
        "value": None,
        "error": str(last_error) if last_error is not None else "all solvers failed to reach optimality",
    }


class SolveTimer:
    """Context manager for timing a single QP solve, for the
    per-control-step computation metric (Research Plan Sec.7.4)."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_seconds = time.perf_counter() - self._t0
        return False

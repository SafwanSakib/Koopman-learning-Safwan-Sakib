"""Baseline #1 -- Model-based MPC with known dynamics.

Research Plan Sec.7.3.1 / Track B Sec.4.2 table row 1: "Use do-mpc directly
with the true ODE -- upper-bound reference, cheapest to implement, do first."

This is the best-case performance ceiling every other method (including our
own) is compared against. No learning, no safety layer beyond whatever
constraint do-mpc supports natively (box constraints on h(x) sublevel sets
are sufficient here since the true model is known exactly).

TODO (Week 6-10):
    - Wrap do-mpc's Model/MPC classes around each sims/*.py true ODE
    - Constrain h(x) >= 0 directly as a do-mpc nonlinear constraint (exact,
      since the model is exact -- no margin/error term needed here, unlike
      the learned-model methods)
    - Return the same result-dict shape as experiments/run_experiment.py
      expects from every baseline
"""

from __future__ import annotations


def run(benchmark: str, config: dict) -> dict:
    """Run this baseline on one benchmark/config; return a results dict
    with the standard fields (tracking error, safety-violation rate, solve
    time, etc. -- see experiments/stats.py for the exact schema once defined).
    """
    raise NotImplementedError("Week 6-10: implement do-mpc wrapper. See module docstring.")

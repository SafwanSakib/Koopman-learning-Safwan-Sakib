"""Baseline #4 -- Linear DeePC around an operating point.

Research Plan Sec.7.3.4 / Track B Sec.4.2 row 4: "Your Week 1-2 toy-system
code, applied around a linearization point of each nonlinear benchmark --
isolates the effect of nonlinearity handling."

TODO (Week 6-10):
    - Reuse control/deepc.py's plain (non-Koopman) DeePCProblem directly.
    - For each benchmark, linearize sims/*.py's dynamics() at a chosen
      operating point (e.g. via finite-difference Jacobian) and generate a
      raw (u, x) trajectory near that point for the Hankel matrices -- do
      NOT lift through models/autoencoder.py for this baseline; that's the
      whole point of the comparison.
"""

from __future__ import annotations


def run(benchmark: str, config: dict) -> dict:
    raise NotImplementedError("Week 6-10: implement linear DeePC (operating-point) baseline.")

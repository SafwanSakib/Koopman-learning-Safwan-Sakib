"""Baseline #7 -- Physics-guided SOS barrier certificate (Lavaei-group
style).

Research Plan Sec.7.3.7 / Track B Sec.4.2 row 7: "Requires a polynomial
approximation of each benchmark's dynamics (e.g. low-order Taylor expansion
or polynomial regression fit) plus an SOS solver (e.g. SumOfSquares.jl via a
Python-Julia bridge, or a Python SOS toolbox if available) -- this is the
most implementation-effort-heavy baseline; start it early within this
window, not last."

Reference: arXiv:2508.01315, "Data-Efficient Control of Polynomial Systems
via Physics-Guided Quadratic Constraints" -- physics-guided + single-
trajectory (fundamental-lemma-style) + robust control barrier certificates,
restricted to polynomial systems solved via sum-of-squares (SOS)
optimization (Research Plan Sec.4.4, row 16 -- "closest four-way collision
found across all search passes").

Isolates the effect of the Koopman-lifted/deep-learning route versus the
SOS/polynomial route on the same physics-informed, data-efficient premise.

TODO (Week 6-10, start early -- flagged as most effort-heavy baseline):
    - Decide on SOS toolchain: SumOfSquares.jl via a Python<->Julia bridge
      (e.g. juliacall/PythonCall), or a pure-Python SOS toolbox if one with
      adequate performance exists -- evaluate both before committing.
    - Fit a low-order polynomial approximation of each benchmark's dynamics
      (Taylor expansion around the operating region, or polynomial
      regression against simulated data).
    - Re-read arXiv:2508.01315 in full before implementing.
"""

from __future__ import annotations


def run(benchmark: str, config: dict) -> dict:
    raise NotImplementedError(
        "Week 6-10 (start early): implement SOS barrier-certificate baseline. "
        "See module docstring -- most implementation-effort-heavy baseline."
    )

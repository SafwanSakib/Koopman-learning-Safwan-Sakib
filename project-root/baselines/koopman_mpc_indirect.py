"""Baseline #2 -- Standard (indirect) Koopman-MPC.

Research Plan Sec.7.3.2 / Track B Sec.4.2 row 2: "Reuse your autoencoder
(without the online DeePC layer) to identify explicit A, B, then standard
linear MPC on the lifted space (Korda & Mezic style)."

This isolates the "identify-then-control" (indirect) pipeline as a baseline
against the paper's "direct" (behavioral, data-matrix) predictor -- see
Theoretical_Background_Full.md Sec.4.4.

TODO (Week 6-10):
    - Train models/autoencoder.py::KoopmanAutoencoder as usual, but at
      control time use the *explicit* learned A, B directly (no DeePC QP,
      no Hankel matrices) -- ordinary linear MPC in z-space, decode to x.
    - Fairness discipline (Track B Sec.4.2): tune MPC horizon/weights with
      comparable effort to the main method.
"""

from __future__ import annotations


def run(benchmark: str, config: dict) -> dict:
    raise NotImplementedError("Week 6-10: implement indirect Koopman-MPC baseline.")

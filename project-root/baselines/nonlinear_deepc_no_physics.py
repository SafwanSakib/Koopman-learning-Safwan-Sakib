"""Baseline #5 / Ablation #9 -- Nonlinear DeePC without physics-informed
regularization (Lian & Jones / Xiong et al. style).

Research Plan Sec.7.3.5 / Track B Sec.4.2 row 5: "Your full method with
lambda_physics = 0 -- nearly free, just a config toggle, but log it as a
fully separate experimental run for clean statistics."

Note the dual role (Track B Sec.4.2 row 9): this SAME configuration is used
both as an independent literature baseline (Lian & Jones 2021 / Xiong et al.
2025 style) and as the ablation isolating the physics-informed loss's
contribution. Track B explicitly flags: "keep this distinction clear in the
results table structure" -- i.e. log/report it under both labels, don't
silently merge the two comparisons.

TODO (Week 6-10):
    - This should NOT require new code beyond a config flag: set
      lambda_physics=0.0 in the models/losses.py total_koopman_loss call and
      otherwise reuse the full pipeline (control/deepc.py +
      control/cbf_constraint.py) unchanged.
"""

from __future__ import annotations


def run(benchmark: str, config: dict) -> dict:
    raise NotImplementedError(
        "Week 6-10: implement by setting lambda_physics=0 on the main pipeline; "
        "see module docstring for the dual baseline/ablation logging requirement."
    )

"""Baseline #6 -- Direct/model-free SACBF (He, Shi, van den Boom, De Schutter
2025 style).

Research Plan Sec.7.3.6 / Track B Sec.4.2 row 6: "Highest-priority external
baseline -- your closest competitor per the lit review. Implement their
state-action CBF formulation as faithfully as possible from the paper;
budget real implementation time here, this is not a quick baseline."

Reference: He et al. (2025), arXiv:2505.15515 -- "From Learning to Safety: A
Direct Data-Driven Framework for Constrained Control." Introduces the
state-action control barrier function (SACBF), a safety certificate
evaluated on state-input pairs directly from transition data, with an
Error-to-State Safety analysis bounding how learning-induced errors shrink
the guaranteed safe set. Structurally: a SINGLE-STEP reactive safety filter
over generic transition data -- no receding-horizon predictive structure, no
Koopman lifting (Introduction_Literature_Review_Draft.md Sec.1.2.8).

This is the single most important external comparison in the paper --
beating/matching it while adding predictive-horizon + physics-informed
capability is the strongest empirical argument for the paper's contribution.

TODO (Week 6-10, budget real time):
    - Re-read arXiv:2505.15515 in full before implementing (not just this
      summary) -- re-verify every claim per the Research Plan Sec.4.4 caveat.
    - Implement the SACBF learned directly from (x, u, x_next) transition
      triples (no lifting map, no autoencoder).
    - Implement their Error-to-State Safety margin computation for the
      apples-to-apples safety-margin comparison against our Theorem 2/3.
"""

from __future__ import annotations


def run(benchmark: str, config: dict) -> dict:
    raise NotImplementedError(
        "Week 6-10: implement SACBF baseline per He et al. 2025. Budget real time -- "
        "see module docstring."
    )

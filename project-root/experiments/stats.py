"""Aggregation and statistical-significance infrastructure.

Track B Implementation Plan Sec.5.1: "Build this before running the full
sweep, not after." Research Plan Sec.7.5: >=10-20 seeds per configuration,
report mean +/- std or 95% CI (not single-run numbers), paired t-test or
Wilcoxon signed-rank wherever the paper claims the method beats a baseline.

TODO (Week 9-14, before the full sweep starts):
    - aggregate_seeds(results: list[dict]) -> dict with mean/std/95%-CI per
      metric, for a fixed (benchmark, method, T, sigma) group
    - paired_significance_test(method_a_results, method_b_results, metric,
      test="wilcoxon" | "ttest") -> p-value, matched by seed
    - Export to the exact tables/figures format paper/ expects (coordinate
      with Track C on this schema before building it, not after)
"""

from __future__ import annotations


def aggregate_seeds(results: list[dict]) -> dict:
    raise NotImplementedError("Week 9-14: implement mean/std/CI aggregation across seeds")


def paired_significance_test(results_a: list[dict], results_b: list[dict],
                              metric: str, test: str = "wilcoxon") -> float:
    """Returns a p-value. `test` in {"wilcoxon", "ttest"}."""
    raise NotImplementedError("Week 9-14: implement paired significance testing")

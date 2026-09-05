"""Sweep orchestration across the full experimental grid.

Track B Implementation Plan Sec.5.1, Sec.5.3: "All 5 benchmarks x 9
baselines/ablations x sweep grid x 10-20 seeds. This is the most
compute-bound phase of the project."

Sweep dimensions (Research Plan Sec.7.2 / Track B Sec.5.1):
    - trajectory length T
    - noise level sigma
    - input-excitation richness (PE order)

TODO (Week 9-14):
    - Build the full config grid (benchmark x method x T x sigma x PE-order
      x seed) from a compact sweep-spec YAML (not 1000s of individual files)
    - Dispatch each point to experiments/run_experiment.py::run_single,
      either in-process or via subprocess/job-array for parallelism
    - If spare compute is available (second machine / cloud burst credits),
      this is the phase to use it -- Track B Sec.5.3 flags it as the
      single most compute-bound step in the whole project.
    - Idempotency: skip already-completed (config, seed) points on rerun
      (important given the compute cost and the likelihood of re-running
      flagged experiments during the Week 19-22 review pass, Track B Sec.7).
"""

from __future__ import annotations


def build_grid(sweep_spec: dict) -> list[dict]:
    """Expand a compact sweep spec into the full list of per-run configs."""
    raise NotImplementedError("Week 9-14")


def dispatch(grid: list[dict], parallel: bool = True) -> None:
    raise NotImplementedError("Week 9-14")

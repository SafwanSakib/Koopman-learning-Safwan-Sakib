"""Single (benchmark, method, T, sigma, seed) experiment runner.

Track B Implementation Plan Sec.5.1. Called by experiments/sweep.py for
every point in the full grid (5 benchmarks x 9 baselines/ablations x sweep
dims x >=10-20 seeds -- Sec.5.3).

Result-dict contract (every method -- our full method AND all 9 baselines/
ablations in baselines/*.py -- must return this shape, so
experiments/stats.py can aggregate uniformly):

    {
        "benchmark": str,
        "method": str,
        "T": int, "sigma": float, "seed": int,
        "rms_tracking_error": float,
        "settling_time": float | None,
        "safety_violation_rate": float,       # percent, Research Plan Sec.7.4
        "min_h": float,                        # min observed h(x_k) over the run
        "T_to_threshold": int | None,          # data efficiency metric
        "solve_time_mean_seconds": float,
        "solve_time_std_seconds": float,
        "h_trajectory": list[float] | None,    # for the theory-vs-practice plot
        "wall_clock_seconds": float,
    }

TODO (Week 9-14):
    - Config loading from experiments/configs/*.yaml
    - Dispatch to the right baselines.*.run() or the main pipeline
      (control/deepc.py + models/autoencoder.py + control/cbf_constraint.py)
      based on `method` in the config
    - Seed everything (numpy, torch, python random) from `seed`
    - Held-out test trajectory: generate separately, never touch during
      training/tuning (Research Plan Sec.7.2)
    - Write result dict to disk (json/parquet) under a path keyed by
      (benchmark, method, T, sigma, seed) for sweep.py / stats.py to collect
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single experiment configuration")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def run_single(config: dict, seed: int) -> dict:
    """Run one (benchmark, method, T, sigma, seed) configuration; return the
    result dict per the module docstring's contract."""
    raise NotImplementedError("Week 9-14: implement dispatch + result-dict construction")


if __name__ == "__main__":
    args = parse_args()
    raise NotImplementedError("Week 9-14: load YAML config, call run_single, write results")

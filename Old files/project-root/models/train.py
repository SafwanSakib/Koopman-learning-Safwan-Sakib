"""Training loop for the physics-informed Koopman autoencoder.

Track B Implementation Plan §2.1, §4.2 row 5/9 (lambda_physics as an ablation
config toggle — do not hardcode it, always read from config).

TODO (Week 2-4):
    - Argparse / YAML config loading (see experiments/configs/)
    - Data loading from a generated trajectory (see sims/*.py data-generation
      functions)
    - Standard train/val split, early stopping on val prediction loss
    - Data-efficiency check (Track B §2.1): train with varying trajectory
      length T at lambda_physics > 0 vs. lambda_physics = 0, plot prediction
      error vs. T for both, confirm physics-informed version reaches a fixed
      error threshold with less data. This is the first empirical check of
      the paper's core "physics prior improves data efficiency" claim.
    - W&B (or structured logging) integration per Research Plan §8

Usage (once implemented):
    python -m models.train --config experiments/configs/vdp_autoencoder.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train physics-informed Koopman autoencoder")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def train(config_path: Path, seed: int) -> dict:
    """Train one autoencoder run; returns a results dict (final losses,
    checkpoint path, etc.) for experiments/stats.py to aggregate."""
    raise NotImplementedError("Week 2-4: implement training loop")


if __name__ == "__main__":
    args = parse_args()
    train(args.config, args.seed)

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
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train physics-informed Koopman autoencoder")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def load_trajectory_data(benchmark: str, data_cfg: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a trajectory and turn it into (x_k, u_k, x_{k+1}) triples --
    one training example per consecutive pair of timesteps."""
    if benchmark == "van_der_pol":
        from sims.van_der_pol import generate_pe_trajectory
        T_samples = data_cfg["T"]
        dt = data_cfg["dt"]
        traj = generate_pe_trajectory(
            x0=np.array([0.5, 0.0]),
            t_span=(0.0, T_samples * dt),
            dt=dt,
            mu=data_cfg["mu"],
            seed=seed,
            excitation=data_cfg["excitation"],
        )
    else:
        raise NotImplementedError(f"load_trajectory_data: benchmark '{benchmark}' not wired in yet")

    x, u = traj["x"], traj["u"]
    if data_cfg.get("noise_sigma", 0.0) > 0:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0, data_cfg["noise_sigma"], size=x.shape)

    x_k = x[:-1]
    x_next = x[1:]
    u_k = u[:-1].reshape(-1, 1)  # (T-1, 1) -- input_dim=1 for Van der Pol
    return x_k, u_k, x_next

def train(config_path: Path, seed: int) -> dict:
    """Train one autoencoder run; returns a results dict (final losses,
    checkpoint path, etc.) for experiments/stats.py to aggregate."""
    import yaml
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    from models.autoencoder import KoopmanAutoencoder
    from models.losses import total_koopman_loss

    with open(config_path) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)

    benchmark = config["benchmark"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # --- Data ---
    x, u, x_next = load_trajectory_data(benchmark, data_cfg, seed)
    n = x.shape[0]
    split = int(n * data_cfg["train_val_split"])

    x_t, u_t, x_next_t = torch.tensor(x, dtype=torch.float32), torch.tensor(u, dtype=torch.float32), torch.tensor(x_next, dtype=torch.float32)
    train_ds = TensorDataset(x_t[:split], u_t[:split], x_next_t[:split])
    val_ds = TensorDataset(x_t[split:], u_t[split:], x_next_t[split:])
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)

    # --- Model ---
    model = KoopmanAutoencoder(
        state_dim=model_cfg["state_dim"],
        input_dim=model_cfg["input_dim"],
        lift_dim=model_cfg["lift_dim"],
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        h_coordinates=[],  # Van der Pol has no safe set yet, see sims/van_der_pol.py
        bilinear=model_cfg["bilinear"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    # --- Training loop with early stopping ---
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(train_cfg["epochs"]):
        model.train()
        for x_batch, u_batch, x_next_batch in train_loader:
            optimizer.zero_grad()
            losses = total_koopman_loss(x_batch, x_next_batch, u_batch, model,
                                         lambda_physics=train_cfg["lambda_physics"],
                                         benchmark=benchmark)
            losses["total"].backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = total_koopman_loss(val_ds.tensors[0], val_ds.tensors[2], val_ds.tensors[1],
                                             model, lambda_physics=train_cfg["lambda_physics"],
                                             benchmark=benchmark)
        val_loss = val_losses["total"].item()
        history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == train_cfg["epochs"] - 1:
            print(f"epoch {epoch}: val_loss={val_loss:.4f} (best={best_val_loss:.4f})")

        if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    # --- Assumption A6 diagnostic (Week 1-2's check_controllability) ---
    from controllers.deepc import check_controllability
    A_np = model.A.detach().numpy()
    B_np = model.B.detach().numpy()
    controllability = check_controllability(A_np, B_np)
    print("controllability check:", controllability)
    eigvals = np.linalg.eigvals(A_np)
    print("eigenvalues of learned A:", eigvals)
    print("eigenvalue magnitudes:", np.abs(eigvals))

    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "controllability": controllability,
        "model": model,
    }


if __name__ == "__main__":
    args = parse_args()
    train(args.config, args.seed)

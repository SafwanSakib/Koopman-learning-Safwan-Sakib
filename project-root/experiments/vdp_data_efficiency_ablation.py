"""Data-efficiency ablation: physics-informed vs. non-physics-informed
Koopman autoencoder training on Van der Pol.

Track B Implementation Plan Sec.2.1: "verify multi-step-ahead prediction
error decreases with more training data, and that the physics-informed
version needs measurably less data to reach a fixed prediction-error
threshold than a no-physics-loss ablation -- this is your first empirical
check of the 'physics prior improves data efficiency' claim central to
the paper."

Protocol:
    1. Generate ONE held-out test trajectory, fixed across the whole
       experiment, never used in training.
    2. For each trajectory length T in a sweep, and for lambda_physics in
       {0.0, 1.0}: train a fresh model on T steps of training data, then
       evaluate multi-step rollout error on the held-out test trajectory.
    3. Report: does the physics-informed curve (lambda_physics=1.0) sit
       below the non-physics curve (lambda_physics=0.0) at every T -- i.e.
       does physics-informed training reach the same accuracy with less
       data?
"""

from __future__ import annotations

import numpy as np
import torch

from models.autoencoder import KoopmanAutoencoder
from models.losses import total_koopman_loss, physics_loss_van_der_pol
from sims.van_der_pol import generate_pe_trajectory
from scipy.stats import wilcoxon


def train_one_model(T: int, lambda_physics: float, seed: int, max_epochs: int = 1000,
                     patience: int = 30) -> KoopmanAutoencoder:
    """Train a single model on T steps of freshly generated Van der Pol
    data, with an internal train/val split and early stopping -- matching
    models/train.py's validated protocol, rather than a fixed epoch count
    (which unfairly under-trains larger-T models relative to what they
    could achieve)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dt = 0.05
    traj = generate_pe_trajectory(x0=np.array([0.5, 0.0]), t_span=(0.0, T * dt), dt=dt,
                                   mu=1.0, seed=seed, excitation="prbs")
    x, u = traj["x"], traj["u"]
    x_k, x_next, u_k = x[:-1], x[1:], u[:-1].reshape(-1, 1)

    split = int(len(x_k) * 0.8)
    x_train, x_val = x_k[:split], x_k[split:]
    u_train, u_val = u_k[:split], u_k[split:]
    xn_train, xn_val = x_next[:split], x_next[split:]

    x_t = torch.tensor(x_train, dtype=torch.float32)
    u_t = torch.tensor(u_train, dtype=torch.float32)
    xn_t = torch.tensor(xn_train, dtype=torch.float32)
    x_v = torch.tensor(x_val, dtype=torch.float32)
    u_v = torch.tensor(u_val, dtype=torch.float32)
    xn_v = torch.tensor(xn_val, dtype=torch.float32)

    model = KoopmanAutoencoder(state_dim=2, input_dim=1, lift_dim=4, hidden_dims=(32, 32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val, best_state, patience_ctr = float("inf"), None, 0
    n_collocation = 64
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        losses = total_koopman_loss(x_t, xn_t, u_t, model, lambda_physics=0.0, benchmark="van_der_pol")
        total = losses["total"]

        if lambda_physics > 0:
            # Physics consistency on the real training batch (as before).
            z_real = model.encode(x_t)
            z_next_real_pred = model.lifted_step(z_real, u_t)
            phys_real = physics_loss_van_der_pol(x_t, u_t, z_next_real_pred, model)

            # Physics consistency on FREE synthetic collocation points --
            # no real simulator data needed, this is the actual mechanism
            # that should buy data efficiency (Theoretical_Background_Full.md
            # Sec.8.1: "even without a data point exactly measuring this").
            # Range chosen from Van der Pol's known limit-cycle amplitude
            # (~2) plus some margin for forced excursions.
            x1_colloc = torch.empty(n_collocation).uniform_(-3.0, 3.0)
            x2_colloc = torch.empty(n_collocation).uniform_(-4.0, 4.0)
            x_colloc = torch.stack([x1_colloc, x2_colloc], dim=-1)
            u_colloc = torch.empty(n_collocation, 1).uniform_(-1.0, 1.0)
            z_colloc = model.encode(x_colloc)
            z_next_colloc_pred = model.lifted_step(z_colloc, u_colloc)
            phys_colloc = physics_loss_van_der_pol(x_colloc, u_colloc, z_next_colloc_pred, model)

            total = total + lambda_physics * (phys_real + phys_colloc)

        total.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = total_koopman_loss(x_v, xn_v, u_v, model, lambda_physics=lambda_physics,
                                           benchmark="van_der_pol")["total"].item()
        if val_loss < best_val:
            best_val, best_state, patience_ctr = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            break

    model.load_state_dict(best_state)
    return model


def evaluate_rollout_error(model: KoopmanAutoencoder, x0_test: np.ndarray,
                            u_test: np.ndarray, x_true_test: np.ndarray) -> float:
    """Multi-step rollout RMSE on the held-out test trajectory."""
    model.eval()
    with torch.no_grad():
        x0_t = torch.tensor(x0_test, dtype=torch.float32).unsqueeze(0)
        u_t = torch.tensor(u_test, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        pred = model.rollout(x0_t, u_t).squeeze(0).numpy()
    return float(np.sqrt(np.mean((pred - x_true_test) ** 2)))


def run_ablation(T_values: list[int], seeds: list[int] = (0, 1, 2, 3, 4)) -> dict:
    dt = 0.05
    test_horizon = 25
    test_traj = generate_pe_trajectory(x0=np.array([0.5, 0.0]), t_span=(0.0, (test_horizon + 1) * dt),
                                        dt=dt, mu=1.0, seed=999, excitation="prbs")  # same x0 as training, different noise realization (seed 999)
    x0_test = test_traj["x"][0]
    u_test = test_traj["u"][:test_horizon]
    x_true_test = test_traj["x"][: test_horizon + 1]

    results = {"physics": {}, "no_physics": {}}
    for T in T_values:
        errs_physics, errs_no_physics = [], []
        for seed in seeds:
            model_physics = train_one_model(T, lambda_physics=1.0, seed=seed)
            model_no_physics = train_one_model(T, lambda_physics=0.0, seed=seed)
            errs_physics.append(evaluate_rollout_error(model_physics, x0_test, u_test, x_true_test))
            errs_no_physics.append(evaluate_rollout_error(model_no_physics, x0_test, u_test, x_true_test))

        mean_p, std_p = float(np.mean(errs_physics)), float(np.std(errs_physics))
        mean_np, std_np = float(np.mean(errs_no_physics)), float(np.std(errs_no_physics))
        results["physics"][T] = {"mean": mean_p, "std": std_p, "raw": errs_physics}
        results["no_physics"][T] = {"mean": mean_np, "std": std_np, "raw": errs_no_physics}
        print(f"T={T}: physics-informed RMSE={mean_p:.4f}+/-{std_p:.4f}, "
              f"no-physics RMSE={mean_np:.4f}+/-{std_np:.4f}  "
              f"({'physics wins' if mean_p < mean_np else 'no-physics wins'})")

        stat, p_value = wilcoxon(errs_physics, errs_no_physics)
        print(f"    Wilcoxon signed-rank test (physics vs no-physics, paired by seed): p={p_value:.4f}")

    all_diffs_physics = [v for T in T_values for v in results["physics"][T]["raw"]]
    all_diffs_no_physics = [v for T in T_values for v in results["no_physics"][T]["raw"]]
    stat, p_pooled = wilcoxon(all_diffs_physics, all_diffs_no_physics)
    n_wins = sum(1 for T in T_values if results["physics"][T]["mean"] < results["no_physics"][T]["mean"])
    print(f"\nPooled across all T (n={len(all_diffs_physics)} paired samples): "
          f"physics-informed wins at {n_wins}/{len(T_values)} data-budget levels, "
          f"pooled Wilcoxon p={p_pooled:.4f}")

    return results


if __name__ == "__main__":
    T_values = [50, 100, 200, 400, 800]
    run_ablation(T_values, seeds=list(range(10)))
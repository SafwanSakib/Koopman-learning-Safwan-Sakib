"""Loss terms for physics-informed Koopman autoencoder training.

Track B Implementation Plan Sec.2.1; Theoretical_Background_Full.md Sec.4.3
(three canonical terms) and Sec.8.2 (physics-informed extension).

    L = reconstruction_loss(x, decoder(encoder(x)))
      + linear_dynamics_loss(encoder(x_next), A @ encoder(x) + B @ u)
      + prediction_loss(x_next, decoder(A @ encoder(x) + B @ u))
      + lambda_physics * physics_loss(...)   # nominal-model or conservation-law term

physics_loss is benchmark-specific (Research Plan Sec.5.2, Track B Sec.2.1):
    - cart-pole: hard-code known kinematics (position integrates velocity
      exactly) into the decoder architecture where possible, per
      Theoretical_Background_Full.md Sec.8.2's "structural prior" guidance,
      rather than purely as a soft loss term.
    - CSTR: energy/mass-balance-consistency term.
    - Van der Pol: nominal-model-agreement term (soft), useful as the primary
      ablation target (lambda_physics=0 baseline, Track B Sec.4.2 row 5/9).

--- ADDED 2026-09-10 per the frozen Track A theory note
(theorem3_margin_bound.tex, "Route 1: empirically validated residual",
Lemma "Validated residual bound", and Assumption A8 "Residual regularity"):
epsilon_phys is NOT just a training-loss number -- Theorem 3's C_3 term
needs a rigorously bounded epsilon_bar, obtained by MEASURING the residual
on a held-out validation set and correcting for how densely that set
covers the operating region (its "fill distance"). See
`held_out_residual_bound` and `fill_distance` below -- these are fully
implemented (not stubs), since they're pure geometry/statistics and don't
depend on the autoencoder itself being trained yet. models/train.py must
call these AFTER every training run and log the resulting bound alongside
the usual loss curves; this is the actual epsilon_bar that feeds Theorem
3's margin (and, at the Track B level, Baseline #6's He et al.-style
margin comparison, and the Robust-Koopman-CBF-SAC-style empirical-margin
idea already in Theoretical_Background_Full.md Sec.7.3).

TODO (Week 2-4):
    - Implement each loss function below
    - Add a `physics_loss_registry` mapping benchmark name -> loss fn, so
      models/train.py can select the right term per benchmark from config
"""

from __future__ import annotations

import numpy as np
import torch


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """||x - decoder(encoder(x))||^2"""
    return torch.mean(torch.sum((x - x_hat) ** 2, dim=-1))


def linear_dynamics_consistency_loss(
    z_next: torch.Tensor, z_next_pred: torch.Tensor
) -> torch.Tensor:
    """||phi(x_{k+1}) - (A phi(x_k) + B u_k)||^2"""
    return torch.mean(torch.sum((z_next - z_next_pred) ** 2, dim=-1))


def prediction_loss(x_next: torch.Tensor, x_next_hat: torch.Tensor) -> torch.Tensor:
    """||x_{k+1} - decoder(A phi(x_k) + B u_k)||^2"""
    return torch.mean(torch.sum((x_next - x_next_hat) ** 2, dim=-1))

def physics_loss_van_der_pol(x: torch.Tensor, u: torch.Tensor, z_next_pred: torch.Tensor,
                              model, dt: float = 0.05, mu: float = 1.0) -> torch.Tensor:
    """Soft nominal-model-agreement term for Van der Pol. Steps the KNOWN
    Van der Pol ODE forward one Euler step from the current (x, u), lifts
    that nominal prediction through the encoder, and penalizes the learned
    dynamics (z_next_pred) for disagreeing with it. This is the primary
    ablation lever: set lambda_physics=0 to reproduce Baseline #5/#9
    (nonlinear DeePC without physics-informed regularization)."""
    x1, x2 = x[:, 0], x[:, 1]
    u_flat = u[:, 0]
    x1_dot = x2
    x2_dot = mu * (1 - x1 ** 2) * x2 - x1 + u_flat
    x_nominal_next = x + dt * torch.stack([x1_dot, x2_dot], dim=-1)

    z_nominal_next = model.encode(x_nominal_next)
    return torch.mean(torch.sum((z_nominal_next - z_next_pred) ** 2, dim=-1))


def physics_loss_cartpole(x: torch.Tensor, u: torch.Tensor, z_next_pred: torch.Tensor,
                           model, dt: float = 0.02, m_c: float = 1.0, m_p: float = 0.1,
                           l: float = 0.5, g: float = 9.81) -> torch.Tensor:
    """Kinematic-consistency term for cart-pole. Mirrors sims/cartpole.py's
    dynamics() exactly, but rewritten with torch ops (batched, autograd-
    compatible) instead of numpy (single-instance, not differentiable) --
    see models/losses.py module docstring's note on why these need to be
    separate implementations of the same equations."""
    p, p_dot, theta, theta_dot = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    u_flat = u[:, 0]
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    total_mass = m_c + m_p

    theta_ddot = (
        g * sin_t - cos_t * (u_flat + m_p * l * theta_dot ** 2 * sin_t) / total_mass
    ) / (l * (4.0 / 3.0 - m_p * cos_t ** 2 / total_mass))
    p_ddot = (u_flat + m_p * l * (theta_dot ** 2 * sin_t - theta_ddot * cos_t)) / total_mass

    x_dot = torch.stack([p_dot, p_ddot, theta_dot, theta_ddot], dim=-1)
    x_nominal_next = x + dt * x_dot

    z_nominal_next = model.encode(x_nominal_next)
    return torch.mean(torch.sum((z_nominal_next - z_next_pred) ** 2, dim=-1))


def physics_loss_cstr(x: torch.Tensor, u: torch.Tensor, z_next_pred: torch.Tensor,
                       model, dt: float = 0.05) -> torch.Tensor:
    """Mass/energy-balance-consistency term for CSTR. Mirrors
    sims/cstr.py's dynamics() exactly, torch-differentiable version,
    using the same DEFAULTS parameters."""
    from sims.cstr import DEFAULTS

    C_A, T = x[:, 0], x[:, 1]
    T_c = u[:, 0]
    p = DEFAULTS
    k = p["k0"] * torch.exp(-p["E_over_R"] / T)
    C_A_dot = (p["q"] / p["V"]) * (p["C_Af"] - C_A) - k * C_A
    T_dot = (
        (p["q"] / p["V"]) * (p["Tf"] - T)
        + (-p["dH"] / (p["rho"] * p["Cp"])) * k * C_A
        + (p["UA"] / (p["V"] * p["rho"] * p["Cp"])) * (T_c - T)
    )
    x_nominal_next = x + dt * torch.stack([C_A_dot, T_dot], dim=-1)

    z_nominal_next = model.encode(x_nominal_next)
    return torch.mean(torch.sum((z_nominal_next - z_next_pred) ** 2, dim=-1))


def physics_loss_quadrotor(x: torch.Tensor, u: torch.Tensor, z_next_pred: torch.Tensor,
                            model, dt: float = 0.02, m: float = 1.0,
                            I: float = 0.01, r: float = 0.25) -> torch.Tensor:
    """Rigid-body / momentum-consistency term for the quadrotor benchmark.
    Mirrors sims/quadrotor.py's dynamics() exactly, torch-differentiable
    version."""
    G_const = 9.81
    _, _, phi, x_dot, z_dot, phi_dot = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
    u1, u2 = u[:, 0], u[:, 1]
    x_ddot = -(u1 + u2) * torch.sin(phi) / m
    z_ddot = (u1 + u2) * torch.cos(phi) / m - G_const
    phi_ddot = (u2 - u1) * r / I

    x_nominal_dot = torch.stack([x_dot, z_dot, phi_dot, x_ddot, z_ddot, phi_ddot], dim=-1)
    x_nominal_next = x + dt * x_nominal_dot

    z_nominal_next = model.encode(x_nominal_next)
    return torch.mean(torch.sum((z_nominal_next - z_next_pred) ** 2, dim=-1))


physics_loss_registry = {
    "van_der_pol": physics_loss_van_der_pol,
    "cartpole": physics_loss_cartpole,
    "cstr": physics_loss_cstr,
    "quadrotor": physics_loss_quadrotor,
    # lorenz intentionally excluded: prediction-accuracy stress test only,
    # no safety constraint / no physics-informed term planned (Research Plan Sec.7.1).
}


def total_koopman_loss(
    x: torch.Tensor,
    x_next: torch.Tensor,
    u: torch.Tensor,
    model,  # models.autoencoder.KoopmanAutoencoder
    lambda_physics: float,
    benchmark: str,
) -> dict[str, torch.Tensor]:
    """Composes all four terms; returns a dict with each component plus
    'total', so training logs can report the breakdown (needed for the
    physics-informed vs. no-physics ablation comparison, Track B Sec.2.1)."""
    out = model.forward(x, u)
    z_next = model.encode(x_next)

    recon = reconstruction_loss(x, out["x_hat"])
    dyn = linear_dynamics_consistency_loss(z_next, out["z_next_pred"])
    pred = prediction_loss(x_next, out["x_next_hat"])

    if lambda_physics > 0:
        physics_fn = physics_loss_registry[benchmark]
        physics = physics_fn(x, u, out["z_next_pred"], model)
    else:
        physics = torch.tensor(0.0)

    total = recon + dyn + pred + lambda_physics * physics
    return {
        "reconstruction": recon,
        "linear_dynamics": dyn,
        "prediction": pred,
        "physics": physics,
        "total": total,
    }


def fill_distance(points: np.ndarray, candidate_points: np.ndarray | None = None) -> float:
    """Estimate the fill distance h_v of `points` over the region of
    interest (Assumption A8 / Lemma "Validated residual bound").

    Fill distance = max over the region of (distance to the NEAREST point
    in `points`). Two modes:

    - `candidate_points` given (recommended): a dense grid or independently
      sampled set covering X x U; returns the true fill-distance estimate
      max_c min_p ||c - p||.
    - `candidate_points=None`: cheap proxy using leave-one-out nearest-
      neighbor distance WITHIN `points` itself (the largest gap between a
      validation point and its nearest neighbor) -- biased low relative to
      the true fill distance (it only sees gaps the validation set itself
      reveals) but requires no extra sampling; fine for an early sanity
      check, should be replaced with the candidate_points mode before
      trusting the resulting Theorem 3 bound for the paper.

    This is real, implemented geometry (not a stub) -- it does not depend
    on the autoencoder being trained, only on having residual-evaluation
    points in hand.
    """
    from scipy.spatial import cKDTree

    points = np.atleast_2d(points)
    if candidate_points is None:
        if points.shape[0] < 2:
            return float("inf")
        tree = cKDTree(points)
        # query 2 nearest (including self), take the 2nd
        dists, _ = tree.query(points, k=2)
        return float(np.max(dists[:, 1]))
    else:
        candidate_points = np.atleast_2d(candidate_points)
        tree = cKDTree(points)
        dists, _ = tree.query(candidate_points, k=1)
        return float(np.max(dists))


def held_out_residual_bound(residual_norms: np.ndarray, L_epsilon: float,
                             fill_dist: float) -> dict:
    """Lemma "Validated residual bound" (theorem3_margin_bound.tex, Route 1):
    given the MEASURED residual norms ||epsilon_phys(x_j, u_j)|| on a
    held-out validation set forming an h_v-net of X x U (h_v = fill_dist,
    see `fill_distance` above), and the Lipschitz constant L_epsilon of
    epsilon_phys (Assumption A8), returns

        epsilon_bar <= epsilon_hat + L_epsilon * h_v,   epsilon_hat = max_j residual_norms[j]

    This is the actual quantity Theorem 3's C_3 term uses -- call this
    after every training run on the held-out validation split (not the
    training split) and log the result alongside the training loss curves.
    Fully implemented (not a stub): pure arithmetic once residuals and a
    fill-distance estimate are in hand.

    Parameters
    ----------
    residual_norms : np.ndarray, shape (M,)
        ||epsilon_phys(x_j, u_j)|| for each of the M held-out validation
        points, i.e. ||z_true_next - (A z + B u)|| evaluated with the
        TRAINED model on data it did not see during training.
    L_epsilon : float
        Lipschitz constant of epsilon_phys (Assumption A8) -- typically
        estimated empirically (e.g. via finite-difference sampling of the
        trained residual field) rather than known a priori; document
        however it was obtained when this is called in practice.
    fill_dist : float
        h_v, from `fill_distance` above.

    Returns
    -------
    dict with keys: "epsilon_hat", "fill_distance", "L_epsilon", "bound".
    """
    residual_norms = np.asarray(residual_norms)
    epsilon_hat = float(np.max(residual_norms))
    bound = epsilon_hat + L_epsilon * fill_dist
    return {
        "epsilon_hat": epsilon_hat,
        "fill_distance": fill_dist,
        "L_epsilon": L_epsilon,
        "bound": bound,
    }

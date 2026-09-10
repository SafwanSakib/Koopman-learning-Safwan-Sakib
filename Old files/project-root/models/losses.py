"""Loss terms for physics-informed Koopman autoencoder training.

Track B Implementation Plan §2.1; Theoretical_Background_Full.md §4.3 (three
canonical terms) and §8.2 (physics-informed extension).

    L = reconstruction_loss(x, decoder(encoder(x)))
      + linear_dynamics_loss(encoder(x_next), A @ encoder(x) + B @ u)
      + prediction_loss(x_next, decoder(A @ encoder(x) + B @ u))
      + lambda_physics * physics_loss(...)   # nominal-model or conservation-law term

physics_loss is benchmark-specific (Research Plan §5.2, Track B §2.1):
    - cart-pole: hard-code known kinematics (position integrates velocity
      exactly) into the decoder architecture where possible, per
      Theoretical_Background_Full.md §8.2's "structural prior" guidance,
      rather than purely as a soft loss term.
    - CSTR: energy/mass-balance-consistency term.
    - Van der Pol: nominal-model-agreement term (soft), useful as the primary
      ablation target (lambda_physics=0 baseline, Track B §4.2 row 5/9).

TODO (Week 2-4):
    - Implement each loss function below
    - Add a `physics_loss_registry` mapping benchmark name -> loss fn, so
      models/train.py can select the right term per benchmark from config
"""

from __future__ import annotations

import torch


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """||x - decoder(encoder(x))||^2"""
    raise NotImplementedError("Week 2-4")


def linear_dynamics_consistency_loss(
    z_next: torch.Tensor, z_next_pred: torch.Tensor
) -> torch.Tensor:
    """||phi(x_{k+1}) - (A phi(x_k) + B u_k)||^2"""
    raise NotImplementedError("Week 2-4")


def prediction_loss(x_next: torch.Tensor, x_next_hat: torch.Tensor) -> torch.Tensor:
    """||x_{k+1} - decoder(A phi(x_k) + B u_k)||^2"""
    raise NotImplementedError("Week 2-4")


def physics_loss_van_der_pol(*args, **kwargs) -> torch.Tensor:
    """Soft nominal-model-agreement term for Van der Pol. This is the primary
    ablation lever: set lambda_physics=0 to reproduce Baseline #5/#9
    (nonlinear DeePC without physics-informed regularization)."""
    raise NotImplementedError("Week 2-4")


def physics_loss_cartpole(*args, **kwargs) -> torch.Tensor:
    """Kinematic-consistency term for cart-pole (soft loss fallback if the
    hard-coded structural prior is not fully wired into the decoder)."""
    raise NotImplementedError("Week 2-4")


def physics_loss_cstr(*args, **kwargs) -> torch.Tensor:
    """Energy/mass-balance-consistency term for CSTR."""
    raise NotImplementedError("Week 2-4")


def physics_loss_quadrotor(*args, **kwargs) -> torch.Tensor:
    """Rigid-body / momentum-consistency term for the quadrotor benchmark."""
    raise NotImplementedError("Week 2-4")


physics_loss_registry = {
    "van_der_pol": physics_loss_van_der_pol,
    "cartpole": physics_loss_cartpole,
    "cstr": physics_loss_cstr,
    "quadrotor": physics_loss_quadrotor,
    # lorenz intentionally excluded: prediction-accuracy stress test only,
    # no safety constraint / no physics-informed term planned (Research Plan §7.1).
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
    physics-informed vs. no-physics ablation comparison, Track B §2.1)."""
    raise NotImplementedError("Week 2-4")

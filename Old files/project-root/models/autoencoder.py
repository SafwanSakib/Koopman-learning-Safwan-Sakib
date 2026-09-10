"""Physics-informed Koopman autoencoder.

Implements Track B Implementation Plan §2.1 / Theoretical_Background_Full.md
§4.3, §8.2.

Architecture contract (do not violate — this is load-bearing for the whole
project, see Research Plan §5.2 and Theoretical_Background_Full.md §6.4):

    - Encoder phi_theta: R^n -> R^N maps raw state x to lifted observable z.
    - Decoder: R^N -> R^n reconstructs x from z.
    - Linear (or bilinear, if a u*z cross term is needed) matrices A, B model
      the lifted dynamics: z_{k+1} ~= A z_k + B u_k.
    - h(x) MUST be embedded as one coordinate of z (i.e. one row of the
      encoder output is exactly the barrier function, or an affine function
      of it). This is what turns the CBF constraint into something linear in
      z, which is what makes the Data-CBF constraint (control/cbf_constraint.py)
      a linear constraint on Hankel-matrix rows instead of a nonlinear one.
      Do not refactor this away for architectural convenience later.

TODO (Week 2-4):
    - Implement KoopmanAutoencoder.forward
    - Implement per-benchmark encoder/decoder configs (van_der_pol, cartpole,
      cstr, quadrotor; lorenz has no h(x), see sims/lorenz.py)
    - Wire h(x) embedding per-benchmark (see sims/*.py safe_set functions)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KoopmanAutoencoder(nn.Module):
    """Physics-informed Koopman lifting: x -> z = phi_theta(x), with linear
    (or bilinear) dynamics A, B learned jointly, and h(x) embedded as one
    coordinate of z.

    Parameters
    ----------
    state_dim : int
        Dimension n of the raw state x.
    input_dim : int
        Dimension m of the control input u.
    lift_dim : int
        Dimension N of the lifted observable z = phi_theta(x). Must be > n.
    hidden_dims : tuple[int, ...]
        Hidden layer widths for encoder/decoder MLPs.
    h_coordinate : int
        Index into z that is constrained to equal (or be affine in) the
        barrier function h(x) for this benchmark. Required, not optional.
    bilinear : bool
        If True, include a u ⊗ z cross term in the dynamics (Koopman
        bilinear realization, cf. Xiong et al. 2025 / Zinage & Bakolas 2023).
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        lift_dim: int,
        hidden_dims: tuple[int, ...] = (64, 64),
        h_coordinate: int = 0,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.lift_dim = lift_dim
        self.h_coordinate = h_coordinate
        self.bilinear = bilinear

        # TODO: build encoder/decoder MLPs from hidden_dims.
        self.encoder: nn.Module | None = None
        self.decoder: nn.Module | None = None

        # Learned lifted-dynamics matrices.
        self.A = nn.Parameter(torch.eye(lift_dim))
        self.B = nn.Parameter(torch.zeros(lift_dim, input_dim))
        if bilinear:
            # One N x N cross-term matrix per input channel: z' += (u_i * C_i) z
            self.C = nn.Parameter(torch.zeros(input_dim, lift_dim, lift_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Week 2-4: implement encoder forward pass")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Week 2-4: implement decoder forward pass")

    def lifted_step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """One-step lifted dynamics prediction z_{k+1} ~= A z_k + B u_k
        (+ bilinear cross term if enabled)."""
        raise NotImplementedError("Week 2-4: implement lifted dynamics step")

    def h_hat(self, z: torch.Tensor) -> torch.Tensor:
        """Extract the barrier-function estimate from a lifted state."""
        return z[..., self.h_coordinate]

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Week 2-4: implement full forward pass")

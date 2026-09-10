"""Physics-informed Koopman autoencoder.

Implements Track B Implementation Plan Sec.2.1 / Theoretical_Background_Full.md
Sec.4.3, Sec.8.2.

Architecture contract (do not violate -- this is load-bearing for the whole
project, see Research Plan Sec.5.2 and Theoretical_Background_Full.md
Sec.6.4, and now the frozen Track A theory note's notation_assumptions.tex):

    - Encoder phi_theta: R^n -> R^N maps raw state x to lifted observable z.
    - Decoder: R^N -> R^n reconstructs x from z.
    - Linear (or bilinear, if a u*z cross term is needed) matrices A, B model
      the lifted dynamics: z_{k+1} = A z_k + B u_k + epsilon_phys(x_k, u_k),
      with ||epsilon_phys|| <= epsilon_bar (a DETERMINISTIC, bounded
      residual -- see models/losses.py's held-out residual bound utility
      for how epsilon_bar is actually measured, per Track A's Lemma
      "Validated residual bound").
    - EACH barrier function h_j MUST be embedded as ONE coordinate of z
      (i.e. one row of the encoder output is exactly h_j(x), or an affine
      function of it): h_j(x) = e_{j}^T Phi(x). This is what turns the CBF
      constraint into something linear in z (controllers/cbf_constraint.py).
      Do not refactor this away for architectural convenience later.

--- UPDATED 2026-09-10 per the frozen Track A theory note: h_coordinate
(singular) is now h_coordinates (PLURAL, list[int]). Track A's Theorem 2
requires h(x) = e_j^T Phi(x) to be a SINGLE linear coordinate per barrier;
benchmarks needing multiple simultaneous bounds (cart-pole: angle bound AND
position bound; CSTR: temperature ceiling AND concentration floor;
quadrotor: one bound per active obstacle) must embed ONE coordinate PER
bound rather than a min()-combined single h -- see
controllers/cbf_constraint.py's module docstring for the full resolution. This
also resolves the open min()-vs-smooth-surrogate question previously
flagged in sims/cartpole.py, sims/cstr.py, sims/quadrotor.py's safe_set()
docstrings: those now return one h_j PER bound, not a combined scalar. ---

Note on Assumption A6 (controllability of the lifted pair (A,B),
notation_assumptions.tex Sec.A6): required for Theorem 1's PE-transfer
result. Run controllers.deepc.check_controllability(model.A.detach().numpy(),
model.B.detach().numpy()) after EVERY training run (Week 2-4), not just
once -- a trained (A,B) is not guaranteed controllable just because the
architecture permits it, and Theorem 1 does not apply if it isn't.

Note on Theorem 1 / PE-transfer (updated 2026-09-05, Consolidated_Findings_
TrackA_TrackB_Revisions.md Sec.2.1, and now confirmed by the frozen theory
note): Shang, Cortes, Zheng (2024, arXiv:2409.16389) already prove
PE-transfer for the case where this lifting is EXACT (epsilon_phys = 0) --
see sims/exact_koopman_toy.py for a worked exact-embedding example and
experiments/pe_transfer_sanity_check.py for the sanity check comparing
this module's (necessarily approximate) learned lifting against that
exact-case anchor. This module's whole reason for existing -- an
approximate, learned, physics-informed lifting with epsilon_phys > 0 -- is
precisely the case Shang et al. do NOT cover, and is Theorem 1's actual
contribution (Theorem "PE-Transfer Under Physics-Informed Lifting").

TODO (Week 2-4):
    - Implement KoopmanAutoencoder.forward
    - Implement per-benchmark encoder/decoder configs (van_der_pol, cartpole,
      cstr, quadrotor; lorenz has no h(x), see sims/lorenz.py)
    - Wire h(x) embedding per-benchmark (see sims/*.py safe_set functions,
      now returning one h_j per bound -- see module docstring above)
    - After training, run controllers.deepc.check_controllability on the
      learned A, B (Assumption A6 diagnostic)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KoopmanAutoencoder(nn.Module):
    """Physics-informed Koopman lifting: x -> z = phi_theta(x), with linear
    (or bilinear) dynamics A, B learned jointly, and each barrier h_j
    embedded as one coordinate of z.

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
    h_coordinates : list[int]
        Indices into z, one per embedded barrier function. Length-1 for
        single-bound benchmarks (Van der Pol, if a bound is used at all);
        length >1 for multi-bound benchmarks (cart-pole, CSTR, quadrotor
        with multiple obstacles) -- see module docstring, "UPDATED
        2026-09-10". Required, not optional; every benchmark with a safe
        set must supply at least one entry.
    bilinear : bool
        If True, include a u (x) z cross term in the dynamics (Koopman
        bilinear realization, cf. Xiong et al. 2025 / Zinage & Bakolas 2023;
        sims/exact_koopman_toy.py is a worked exact example of this form).
    """
    def _build_mlp(self, in_dim: int, out_dim: int, hidden_dims: tuple[int, ...]) -> nn.Module:
        """Small helper: builds Linear -> activation -> ... -> Linear."""
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())  # smooth activation, common choice for dynamical-systems autoencoders
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))  # final layer: no activation (raw output)
        return nn.Sequential(*layers)

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        lift_dim: int,
        hidden_dims: tuple[int, ...] = (64, 64),
        h_coordinates: list[int] | None = None,
        bilinear: bool = False,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.lift_dim = lift_dim
        self.h_coordinates = h_coordinates or []
        self.bilinear = bilinear
        # State normalization (needed when different state dimensions live
        # on very different physical scales, e.g. CSTR's C_A~1 vs T~300 --
        # a raw MLP struggles badly with that scale mismatch. Defaults to
        # no-op (mean=0, std=1) for benchmarks that don't need it, like
        # Van der Pol and cart-pole, whose state dimensions are all
        # roughly the same order of magnitude already.
        if state_mean is None:
            state_mean = torch.zeros(state_dim)
        if state_std is None:
            state_std = torch.ones(state_dim)
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std)

        # TODO: build encoder/decoder MLPs from hidden_dims.
        self.encoder = self._build_mlp(state_dim, lift_dim, hidden_dims)
        self.decoder = self._build_mlp(lift_dim, state_dim, hidden_dims)

        # Learned lifted-dynamics matrices.
        self.A = nn.Parameter(torch.eye(lift_dim) + 0.01 * torch.randn(lift_dim, lift_dim))
        self.B = nn.Parameter(torch.zeros(lift_dim, input_dim))
        if bilinear:
            # One N x N cross-term matrix per input channel: z' += (u_i * C_i) z
            self.C = nn.Parameter(torch.zeros(input_dim, lift_dim, lift_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.state_mean) / self.state_std
        return self.encoder(x_norm)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_norm = self.decoder(z)
        return x_norm * self.state_std + self.state_mean

    def lifted_step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """One-step lifted dynamics prediction z_{k+1} ~= A z_k + B u_k
        (+ bilinear cross term if enabled)."""
        z_next = z @ self.A.T + u @ self.B.T
        if self.bilinear:
            # u: (batch, input_dim), z: (batch, lift_dim), C: (input_dim, lift_dim, lift_dim)
            # cross term: sum over input channels i of u_i * (C_i @ z)
            cross = torch.einsum("bi,ijk,bk->bj", u, self.C, z)
            z_next = z_next + cross
        return z_next

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        z_next_pred = self.lifted_step(z, u)
        x_next_hat = self.decode(z_next_pred)
        return {
            "z": z,
            "x_hat": x_hat,
            "z_next_pred": z_next_pred,
            "x_next_hat": x_next_hat,
        }

    def h_hat(self, z: torch.Tensor) -> torch.Tensor:
        """Extract ALL embedded barrier-function estimates from a lifted
        state, as a vector (one entry per self.h_coordinates). For the
        common single-bound case this is a length-1 tensor; index [0] for
        a scalar. Renamed conceptually from a scalar-returning method
        (pre-2026-09-10) to reflect multi-barrier support -- see module
        docstring."""
        if not self.h_coordinates:
            raise ValueError(
                "h_coordinates is empty -- this benchmark's safe set (if any) "
                "was not wired in. See sims/*.py safe_set() and Track B README."
            )
        return z[..., self.h_coordinates]


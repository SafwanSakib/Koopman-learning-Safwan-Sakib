"""Mass-spring-damper: toy LTI system for validating the plain (non-Koopman,
non-CBF) DeePC QP against classical LQR.

Track B Implementation Plan Sec.1.2: "Implement the DeePC QP
(Theoretical_Background_Full.md Sec.3.2-3.3) on a mass-spring-damper -- no
Koopman, no CBF yet. This validates your core QP machinery (Hankel matrix
construction, u_ini/y_ini handling, regularization) in the simplest
possible setting."

Continuous-time dynamics:
    m x_ddot + c x_dot + k x = u
State x = [position, velocity]. Output y = x (full state feedback -- the
simplest valid case for behavioral theory, and it makes the DeePC-vs-LQR
comparison a direct state-trajectory comparison rather than needing an
observer).

    x_dot = A_c x + B_c u,   A_c = [[0, 1], [-k/m, -c/m]],  B_c = [[0], [1/m]]
    y = x

Discretized via zero-order hold (scipy.signal.cont2discrete) at sample time
dt, giving x_{k+1} = A x_k + B u_k -- this is the EXACT discrete-time
system DeePC's Fundamental-Lemma machinery assumes, so it's the correct
comparison target for classical discrete-time LQR (not a Euler
approximation, which would bias the comparison).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import cont2discrete


def continuous_matrices(m: float = 1.0, c: float = 0.4, k: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    A_c = np.array([[0.0, 1.0], [-k / m, -c / m]])
    B_c = np.array([[0.0], [1.0 / m]])
    return A_c, B_c


def discrete_matrices(dt: float, m: float = 1.0, c: float = 0.4, k: float = 2.0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretization at sample time dt."""
    A_c, B_c = continuous_matrices(m, c, k)
    C = np.eye(2)
    D = np.zeros((2, 1))
    A_d, B_d, C_d, D_d, _ = cont2discrete((A_c, B_c, C, D), dt, method="zoh")
    return A_d, B_d


def simulate_discrete(A: np.ndarray, B: np.ndarray, x0: np.ndarray,
                       u_sequence: np.ndarray) -> np.ndarray:
    """Roll out x_{k+1} = A x_k + B u_k for the given input sequence.

    Parameters
    ----------
    u_sequence : np.ndarray, shape (T,) or (T, 1)

    Returns
    -------
    np.ndarray, shape (T+1, n) -- includes x0 as the first row.
    """
    u_sequence = np.atleast_1d(u_sequence).reshape(-1, 1)
    T = u_sequence.shape[0]
    n = A.shape[0]
    x = np.zeros((T + 1, n))
    x[0] = x0
    for k in range(T):
        x[k + 1] = A @ x[k] + (B @ u_sequence[k])
    return x


def generate_pe_input(T: int, seed: int = 0, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """i.i.d. uniform random input sequence. Generically persistently
    exciting of any fixed finite order almost surely (the resulting Hankel
    matrix is full rank with probability 1 for a continuous input
    distribution) -- sufficient for this toy-system validation; the full
    benchmark simulators (sims/van_der_pol.py etc.) will eventually use a
    more deliberate PRBS/multisine scheme (see their generate_pe_trajectory
    TODOs), but random noise is the standard, simplest choice for a linear
    toy system and is what most DeePC tutorials/papers use for exactly this
    kind of sanity check.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=T)

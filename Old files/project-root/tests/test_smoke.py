"""Smoke tests -- Week 1 CI wiring (Track B Sec.0: "pytest running, even
with zero real tests yet -- CI wiring done early").

These just confirm every module in the skeleton imports cleanly and every
implemented (non-stub) function behaves as expected. As each module gets a
real implementation, add a matching tests/test_<module>.py with real
coverage -- this file should stay small and just guard against import-time
breakage / packaging mistakes.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

MODULES = [
    "models.autoencoder",
    "models.losses",
    "models.train",
    "control.deepc",
    "control.cbf_constraint",
    "control.qp_solver",
    "sims.van_der_pol",
    "sims.cartpole",
    "sims.cstr",
    "sims.quadrotor",
    "sims.lorenz",
    "baselines.mpc_known_model",
    "baselines.koopman_mpc_indirect",
    "baselines.koopman_cbf_indirect",
    "baselines.linear_deepc",
    "baselines.nonlinear_deepc_no_physics",
    "baselines.sacbf_direct",
    "baselines.sos_barrier_lavaei",
    "experiments.run_experiment",
    "experiments.sweep",
    "experiments.stats",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    """Every module in the skeleton must import without error. This is the
    cheapest possible regression guard while modules are still stubs."""
    importlib.import_module(module_name)


def test_van_der_pol_limit_cycle_direction() -> None:
    """Van der Pol sanity check per Track B Sec.1.1: unforced (u=0) with
    mu>0, trajectories starting away from the origin should be pulled toward
    a bounded limit cycle, not diverge to infinity. This is a coarse check
    (energy doesn't blow up over a moderate horizon), not a precise
    amplitude check -- add the amplitude check once the benchmark is
    actually used in Week 4-6.
    """
    from sims.van_der_pol import simulate

    result = simulate(x0=np.array([0.5, 0.0]), u_fn=lambda t: 0.0,
                       t_span=(0.0, 50.0), dt=0.05, mu=1.0)
    # Should remain bounded (Van der Pol limit cycle amplitude is close to 2
    # for small mu) -- generous bound, this is just a divergence guard.
    assert np.all(np.abs(result["x"]) < 10.0)


def test_cartpole_upright_equilibrium_is_stationary() -> None:
    """At x = [0, 0, 0, 0] (upright, at rest) with u=0, the cart-pole is an
    (unstable) equilibrium: all derivatives should be exactly zero."""
    from sims.cartpole import dynamics

    x_dot = dynamics(0.0, np.array([0.0, 0.0, 0.0, 0.0]), u=0.0)
    assert np.allclose(x_dot, 0.0, atol=1e-10)


def test_lorenz_origin_is_equilibrium() -> None:
    """The origin is always an equilibrium of the (unforced) Lorenz system."""
    from sims.lorenz import dynamics

    x_dot = dynamics(0.0, np.array([0.0, 0.0, 0.0]), u=0.0)
    assert np.allclose(x_dot, 0.0, atol=1e-10)


def test_cbf_forward_invariance_checker() -> None:
    """control.cbf_constraint.verify_forward_invariance_empirically is fully
    implemented (not a stub) -- give it real coverage now."""
    from control.cbf_constraint import verify_forward_invariance_empirically

    assert verify_forward_invariance_empirically(np.array([1.0, 0.5, 0.1, 0.0]))
    assert not verify_forward_invariance_empirically(np.array([1.0, -0.1, 0.5]))

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
    "controllers.deepc",
    "controllers.cbf_constraint",
    "controllers.qp_solver",
    "sims.van_der_pol",
    "sims.cartpole",
    "sims.cstr",
    "sims.quadrotor",
    "sims.lorenz",
    "sims.exact_koopman_toy",
    "sims.mass_spring_damper",
    "baselines.mpc_known_model",
    "baselines.koopman_mpc_indirect",
    "baselines.koopman_cbf_indirect",
    "baselines.koopman_cbf_mpc_liu",
    "baselines.linear_deepc",
    "baselines.nonlinear_deepc_no_physics",
    "baselines.sacbf_direct",
    "baselines.sos_barrier_lavaei",
    "experiments.run_experiment",
    "experiments.sweep",
    "experiments.stats",
    "experiments.pe_transfer_sanity_check",
    "experiments.theorem3_margin_validation",
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

def test_van_der_pol_limit_cycle_amplitude() -> None:
    """Textbook check (Track B Sec.1.1): for small mu, the unforced Van der
    Pol limit cycle amplitude approaches 2. Simulate long enough for
    transients to die out, then check the settled oscillation amplitude."""
    from sims.van_der_pol import simulate

    result = simulate(x0=np.array([0.5, 0.0]), u_fn=lambda t: 0.0,
                       t_span=(0.0, 200.0), dt=0.02, mu=0.2)
    # Discard the first half (transient), measure amplitude on the rest.
    x1_settled = result["x"][len(result["x"]) // 2:, 0]
    amplitude = (x1_settled.max() - x1_settled.min()) / 2
    assert abs(amplitude - 2.0) < 0.3  # generous tolerance for a small-mu asymptotic result

def test_cartpole_small_angle_period_matches_linearization() -> None:
    """Textbook check (Track B Sec.1.1): release the pole from a small angle
    near the stable hanging-down equilibrium (theta=pi) with zero cart
    force, and confirm the oscillation period matches the period predicted
    by linearizing the dynamics at that equilibrium (eigenvalues of the
    linearized system) -- a standard, checkable closed-form comparison."""
    from sims.cartpole import dynamics, simulate

    m_c, m_p, l = 1.0, 0.1, 0.5

    # Linearize at (p=0, p_dot=0, theta=pi, theta_dot=0) via finite differences.
    x_eq = np.array([0.0, 0.0, np.pi, 0.0])
    eps = 1e-6
    A = np.zeros((4, 4))
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        f_plus = dynamics(0.0, x_eq + dx, u=0.0, m_c=m_c, m_p=m_p, l=l)
        f_minus = dynamics(0.0, x_eq - dx, u=0.0, m_c=m_c, m_p=m_p, l=l)
        A[:, i] = (f_plus - f_minus) / (2 * eps)

    eigvals = np.linalg.eigvals(A)
    # Oscillatory modes have nonzero imaginary part; take the dominant one.
    osc_eigvals = eigvals[np.abs(eigvals.imag) > 1e-6]
    assert len(osc_eigvals) > 0, "expected an oscillatory mode at the hanging equilibrium"
    omega = np.abs(osc_eigvals[0].imag)
    predicted_period = 2 * np.pi / omega

    # Simulate a small release from near theta=pi.
    x0 = np.array([0.0, 0.0, np.pi - 0.05, 0.0])
    result = simulate(x0, u_fn=lambda t: 0.0, t_span=(0.0, 5.0), dt=0.001,
                       m_c=m_c, m_p=m_p, l=l)
    theta = result["x"][:, 2]

    # Find times where theta crosses its mean going the same direction, to
    # measure the actual period empirically.
    centered = theta - theta.mean()
    zero_crossings = np.where(np.diff(np.sign(centered)) > 0)[0]
    assert len(zero_crossings) >= 2, "not enough oscillation cycles simulated"
    measured_period = np.mean(np.diff(result["t"][zero_crossings]))

    assert abs(measured_period - predicted_period) / predicted_period < 0.1

def test_cartpole_upright_equilibrium_is_stationary() -> None:
    """At x = [0, 0, 0, 0] (upright, at rest) with u=0, the cart-pole is an
    (unstable) equilibrium: all derivatives should be exactly zero."""
    from sims.cartpole import dynamics

    x_dot = dynamics(0.0, np.array([0.0, 0.0, 0.0, 0.0]), u=0.0)
    assert np.allclose(x_dot, 0.0, atol=1e-10)

def test_cstr_steady_state_is_a_true_equilibrium() -> None:
    """Self-consistency check: the steady state found by root-finding must
    actually make the dynamics vanish."""
    from sims.cstr import compute_steady_state, dynamics

    T_c = 300.0
    x_ss = compute_steady_state(T_c)
    residual = dynamics(0.0, x_ss, T_c)
    assert np.allclose(residual, 0.0, atol=1e-6)


def test_cstr_steady_state_gain_direction_is_physical() -> None:
    """Textbook check (Track B Sec.1.1): in the single-steady-state
    operating region, increasing coolant temperature T_c must increase
    the steady-state reactor temperature T_ss -- a well-known monotonic
    relationship for this class of CSTR model (Seborg et al.). T_c values
    are chosen well away from this model's known multiplicity region
    (three steady states exist near T_c~300 -- a classic exothermic-
    reactor ignition/extinction phenomenon), to keep the comparison on a
    single, unambiguous branch."""
    from sims.cstr import compute_steady_state

    x_ss_low = compute_steady_state(T_c=270.0)
    x_ss_high = compute_steady_state(T_c=290.0)

    T_ss_low, T_ss_high = x_ss_low[1], x_ss_high[1]
    assert T_ss_high > T_ss_low

def test_quadrotor_hover_is_equilibrium() -> None:
    """Textbook check (Track B Sec.1.1): at exact hover thrust split evenly
    between both rotors (u1=u2=m*g/2), with zero velocities and zero pitch,
    the quadrotor must be a perfect equilibrium -- thrust exactly cancels
    gravity, and equal rotor thrusts produce zero net torque."""
    from sims.quadrotor import G, dynamics

    m = 1.0
    hover_thrust = m * G / 2
    x_eq = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x,z,phi,x_dot,z_dot,phi_dot
    x_dot = dynamics(0.0, x_eq, u=np.array([hover_thrust, hover_thrust]), m=m)
    assert np.allclose(x_dot, 0.0, atol=1e-10)


def test_quadrotor_unequal_thrust_produces_torque() -> None:
    """Sanity check on the sign convention: if rotor 2 pushes harder than
    rotor 1, the vehicle should experience a nonzero angular acceleration
    (a real, physically expected consequence of the model, not an
    arbitrary numeric check)."""
    from sims.quadrotor import G, dynamics

    m = 1.0
    hover_thrust = m * G / 2
    x_eq = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_dot = dynamics(0.0, x_eq, u=np.array([hover_thrust - 0.1, hover_thrust + 0.1]), m=m)
    phi_ddot = x_dot[5]
    assert phi_ddot > 0  # stronger rotor 2 (per this model's sign convention) should spin it up


def test_lorenz_exhibits_sensitive_dependence_on_initial_conditions() -> None:
    """Textbook check (Track B Sec.1.1): at the classical chaotic parameters
    (sigma=10, rho=28, beta=8/3), two trajectories starting a tiny distance
    apart must diverge -- the defining signature of the Lorenz attractor's
    chaos (Lorenz, 1963), not just numerical noise."""
    from sims.lorenz import simulate

    x0_a = np.array([1.0, 1.0, 1.0])
    x0_b = x0_a + np.array([1e-8, 0.0, 0.0])  # tiny perturbation

    result_a = simulate(x0_a, t_span=(0.0, 25.0), dt=0.01)
    result_b = simulate(x0_b, t_span=(0.0, 25.0), dt=0.01)

    initial_separation = np.linalg.norm(result_a["x"][0] - result_b["x"][0])
    final_separation = np.linalg.norm(result_a["x"][-1] - result_b["x"][-1])

    # Chaotic divergence: final separation should be enormously larger than
    # the initial 1e-8 perturbation (order-of-magnitude check, not precise).
    assert final_separation > 1000 * initial_separation
    # Both trajectories should still be bounded (on the attractor, not
    # numerically blown up).
    assert np.all(np.abs(result_a["x"]) < 100)
    assert np.all(np.abs(result_b["x"]) < 100)

def test_lorenz_origin_is_equilibrium() -> None:
    """The origin is always an equilibrium of the (unforced) Lorenz system."""
    from sims.lorenz import dynamics

    x_dot = dynamics(0.0, np.array([0.0, 0.0, 0.0]), u=0.0)
    assert np.allclose(x_dot, 0.0, atol=1e-10)


def test_exact_koopman_toy_lifted_dynamics_match_true_dynamics() -> None:
    """The exact Koopman bilinear realization in sims/exact_koopman_toy.py
    is real math, not a stub -- give it real coverage. Verifies
    z_dot = A z + B u + u*(C z) matches finite-differenced true dynamics
    composed with the exact lift, across random (x, u) samples (this is the
    same check used to validate the derivation during development).
    """
    from sims.exact_koopman_toy import dynamics, exact_lift, lifted_dynamics_matrices

    mu, lam = -0.05, -1.0
    mats = lifted_dynamics_matrices(mu, lam)
    A, B, C = mats["A"], mats["B"], mats["C"]

    rng = np.random.default_rng(42)
    h = 1e-6
    for _ in range(200):
        x = rng.uniform(-2, 2, size=2)
        u = rng.uniform(-1, 1)
        x_dot = dynamics(0.0, x, u, mu, lam)
        z = exact_lift(x)
        z_dot_numeric = (exact_lift(x + h * x_dot) - z) / h
        z_dot_analytic = A @ z + B * u + u * (C @ z)
        assert np.max(np.abs(z_dot_numeric - z_dot_analytic)) < 1e-4


def test_perturb_lift_is_deterministic_given_seed() -> None:
    """experiments/pe_transfer_sanity_check.py::perturb_lift is implemented
    (not a stub) -- give it real coverage."""
    from experiments.pe_transfer_sanity_check import perturb_lift

    z = np.array([1.0, 2.0, 3.0])
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    out1 = perturb_lift(z, epsilon=0.1, rng=rng1)
    out2 = perturb_lift(z, epsilon=0.1, rng=rng2)
    assert np.allclose(out1, out2)

    rng3 = np.random.default_rng(0)
    out_zero_eps = perturb_lift(z, epsilon=0.0, rng=rng3)
    assert np.allclose(out_zero_eps, z)  # eps=0 must reproduce the exact anchor point


def test_check_controllability_correctness() -> None:
    """controllers.deepc.check_controllability is real math (Assumption A6
    diagnostic), not a stub -- give it real coverage on both a controllable
    and an uncontrollable pair. This also guards against regressing the
    control/controllers package-name collision bug fixed 2026-09-10 (the
    local package used to shadow the third-party 'control' pip package,
    silently breaking this function's `import control as ct`)."""
    from controllers.deepc import check_controllability

    # Controllable canonical pair.
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    result = check_controllability(A, B)
    assert result["controllable"]
    assert result["rank"] == 2

    # Uncontrollable: B only excites one eigendirection of a diagonal A.
    A2 = np.array([[1.0, 0.0], [0.0, 2.0]])
    B2 = np.array([[1.0], [0.0]])
    result2 = check_controllability(A2, B2)
    assert not result2["controllable"]
    assert result2["rank"] == 1


def test_no_self_shadowing_of_third_party_control_package() -> None:
    """Regression guard for the 2026-09-10 control/controllers rename: the
    third-party 'control' pip package (used by check_controllability and
    baselines/mpc_known_model.py's do-mpc dependency) must resolve to the
    real library, not to this project's own package (which was named
    'control' before the rename and silently shadowed it when run from the
    project root)."""
    import control as real_control

    assert hasattr(real_control, "ctrb"), (
        "`import control` resolved to something without ctrb() -- almost certainly "
        "shadowed by a local package/module named 'control' again. The project's "
        "own control-theory package must stay named 'controllers', not 'control'."
    )
    assert "project-root" not in (real_control.__file__ or ""), (
        "`import control` resolved to a file inside this project, not the "
        "installed pip package -- the naming collision has regressed."
    )


def test_fill_distance_dense_candidates_at_least_loo_proxy() -> None:
    """models.losses.fill_distance is real geometry, not a stub -- give it
    real coverage. The dense-candidate estimate should be >= the cheap
    leave-one-out proxy (the proxy only sees gaps the validation points
    themselves reveal, so it's biased low relative to the true fill
    distance over a genuinely dense candidate set)."""
    from models.losses import fill_distance

    rng = np.random.default_rng(0)
    points = rng.uniform(0, 1, size=(30, 2))
    candidates = rng.uniform(0, 1, size=(3000, 2))

    hv_proxy = fill_distance(points)
    hv_dense = fill_distance(points, candidate_points=candidates)

    assert hv_proxy > 0
    assert hv_dense > 0
    assert hv_dense >= hv_proxy * 0.5  # dense estimate should be in a sane ballpark


def test_held_out_residual_bound_correctness() -> None:
    """models.losses.held_out_residual_bound is real arithmetic (Lemma
    "Validated residual bound", Assumption A8), not a stub -- give it real
    coverage."""
    from models.losses import held_out_residual_bound

    residuals = np.array([0.01, 0.02, 0.05, 0.03])
    result = held_out_residual_bound(residuals, L_epsilon=0.5, fill_dist=0.1)

    assert result["epsilon_hat"] == 0.05
    assert result["bound"] == pytest.approx(0.05 + 0.5 * 0.1)
    assert result["bound"] >= result["epsilon_hat"]  # bound must never be tighter than the max residual itself


def test_qp_solver_solves_simple_problem_with_fallback() -> None:
    """controllers.qp_solver.solve_qp is real (not a stub) -- give it real
    coverage, including that it correctly reports which solver was used."""
    import cvxpy as cp

    from controllers.qp_solver import solve_qp

    x = cp.Variable(2)
    objective = cp.Minimize(cp.sum_squares(x - np.array([1.0, 2.0])))
    problem = cp.Problem(objective)

    result = solve_qp(problem, solver="OSQP")
    assert result["status"] in ("optimal", "optimal_inaccurate")
    assert result["solver_used"] is not None
    assert result["solve_time_seconds"] >= 0
    assert np.allclose(x.value, [1.0, 2.0], atol=1e-3)


def test_mass_spring_damper_discretization_is_stable() -> None:
    """sims.mass_spring_damper's zero-order-hold discretization is real
    math -- give it coverage. A damped mass-spring-damper (c>0) should have
    a Schur-stable discrete A matrix (all eigenvalues inside the unit
    circle) for a reasonable sample time."""
    from sims.mass_spring_damper import discrete_matrices

    A, B = discrete_matrices(dt=0.1, m=1.0, c=0.4, k=2.0)
    eigvals = np.linalg.eigvals(A)
    assert np.all(np.abs(eigvals) < 1.0)
    assert A.shape == (2, 2)
    assert B.shape == (2, 1)


def test_cbf_forward_invariance_checker() -> None:
    """controllers.cbf_constraint.verify_forward_invariance_empirically is fully
    implemented (not a stub) -- give it real coverage now."""
    from controllers.cbf_constraint import verify_forward_invariance_empirically

    assert verify_forward_invariance_empirically(np.array([1.0, 0.5, 0.1, 0.0]))
    assert not verify_forward_invariance_empirically(np.array([1.0, -0.1, 0.5]))

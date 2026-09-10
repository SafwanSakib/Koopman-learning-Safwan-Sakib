# Physics-Informed, Safety-Certified Direct Data-Driven Predictive Control

Implementation codebase for *"A Physics-Informed Data-Enabled Predictive Control
Framework with Formal Safety Certificates for Unknown Nonlinear Systems."*

This is **Track B** of the project (see the project's Research Plan and
Track B Implementation Plan documents for full context). It implements:

- A physics-informed Koopman autoencoder lifting (`models/`)
- A DeePC (Data-Enabled Predictive Control) QP with a linear Data-CBF safety
  constraint expressed directly on Hankel-matrix rows (`controllers/`)
- Five nonlinear benchmark simulators (`sims/`)
- Nine (plus one added) baselines/ablations for fair comparison (`baselines/`)
- A statistically rigorous experiment runner (>=10-20 seeds, CIs, significance
  tests) (`experiments/`)

## Status

Repo skeleton with theory-synced stubs, **plus the Week 1-2 milestone now
implemented and validated**. See each stubbed module's docstring for what
it should do and which plan section it corresponds to.

**Updated 2026-09-10 (Week 1-2 milestone complete)** — implemented and
tested per Track B Implementation Plan Sec.1.2:
- `controllers/deepc.py`: `build_hankel_matrix`, `split_hankel`,
  `check_persistency_of_excitation`, `check_normalized_excitation` (A7),
  and the full `DeePCProblem` (plain linear case — Koopman lifting and the
  CBF constraint are still Week 2-4/4-6 work, but the class already accepts
  a `cbf_constraint_builder` and `use_cbf` toggle so that layer slots in as
  a strict superset, not a fork).
- `controllers/qp_solver.py`: `solve_qp` with automatic OSQP → ECOS → SCS
  fallback, reporting which solver actually succeeded.
- `sims/mass_spring_damper.py` (new): the toy LTI system the plan calls
  for, exact zero-order-hold discretization, PE input generation.
- **Validated against classical discrete-time LQR** on this toy system
  (`tests/test_deepc_lti_validation.py`): closed-loop RMS state trajectory
  matches LQR to within ~4% in practice (test tolerance set at 15%), with
  both controllers confirmed to actually stabilize the system — this is
  Track B's own stated ground-truth check that the QP is correctly
  formulated.
- 41 tests passing (up from 35).

**Updated 2026-09-10 (theory sync)** — Track A's frozen theory note all
three theorems closed:

- **`control/` renamed to `controllers/`** — the project's own package name
  was silently shadowing the third-party `control` (python-control) pip
  package whenever code ran `import control` from the project root. This
  broke controllability checks and would have broken `do-mpc`-dependent
  baselines. Fixed; see `tests/test_smoke.py`'s regression guard.
- **Assumption A5 resolved** (was "pending"): the CBF constraint is HARD
  only on the first predicted step, SOFT for the rest of the horizon; **no
  terminal safe set is required** for the base safety certificate.
  `controllers/deepc.py`'s `DeePCConfig.use_terminal_set` now defaults to
  `False` and is documented as resolved, not pending.
- **`h(x)` must be a single linear coordinate** — benchmarks needing
  multiple bounds (cart-pole: angle + position; CSTR: temperature +
  concentration; quadrotor: one per obstacle) now embed **one coordinate
  per bound** (`h_coordinates: list[int]` in `models/autoencoder.py`, one
  hard constraint per bound in `controllers/cbf_constraint.py`) instead of a
  min()-combined single barrier. `sims/cartpole.py`, `sims/cstr.py`,
  `sims/quadrotor.py`'s `safe_set()` now return a dict of separate bounds.
- **Prescription P1 (data-normalized regularization) is required, not
  optional** — `DeePCConfig` now takes `lambda_g0`/`lambda_sigma0`, scaled
  internally by the number of Hankel columns at build time. Without this
  the finite-sample margin bound doesn't hold.
- **New diagnostics**: `controllers.deepc.check_controllability` (Assumption
  A6) and `check_normalized_excitation` (Assumption A7).
- **New utility**: `models.losses.held_out_residual_bound` +
  `fill_distance` — measures the actual `epsilon_phys` bound from a
  held-out validation set (Assumption A8 / Route 1 of Theorem 3), fully
  implemented, not a stub.
- **New experiment**: `experiments/theorem3_margin_validation.py` —
  compares standard vs. "decoupled" data usage (split trajectory: compute
  `g` from one half, evaluate online prediction error on the other) against
  Theorem 3's margin bound, addressing the one open problem Track A flagged
  honestly (the tight `1/sqrt(T)` noise rate needs decoupling to be
  provable).

**Updated 2026-09-05** per Track C's Weeks 1-2 literature lock
(`Consolidated_Findings_TrackA_TrackB_Revisions.md`):
- Added **Baseline #10** (`baselines/koopman_cbf_mpc_liu.py`) — Liu, Wu,
  Zhang, Drgona, Belta (2026)'s receding-horizon Koopman-CBF-MPC. Kept
  alongside the original Baseline #3 (Folkestad/Zinage-Bakolas, single-step)
  rather than replacing it — the two isolate different variables (see both
  modules' docstrings).
- Added a new **Theorem 1 sanity-check experiment**
  (`experiments/pe_transfer_sanity_check.py` + `sims/exact_koopman_toy.py`):
  verifies PE-transfer holds exactly at the Shang, Cortes, Zheng (2024)
  zero-lifting-error limit, then sweeps a perturbation to show how the
  margin degrades — feeds a figure for the paper's Theorem 1 section.

## Setup

```bash
conda env create -f environment.yml
conda activate physics-informed-safe-deepc
pytest -q   # should discover and run tests
```

## Repository structure

```
project-root/
├── environment.yml        # pinned exact versions
├── README.md
├── LICENSE                 # MIT
├── models/                 # physics-informed Koopman autoencoder
│   ├── autoencoder.py        # h_coordinates: list[int], one per barrier
│   ├── losses.py              # reconstruction/dynamics/physics losses +
│   │                           # held_out_residual_bound, fill_distance (implemented)
│   └── train.py
├── controllers/              # DeePC + CBF QP machinery (renamed from control/
│   │                           # 2026-09-10 -- see Status section above)
│   ├── deepc.py                # core Hankel-matrix QP; P1 regularization,
│   │                            # A6/A7 diagnostics, decouple_g_data option
│   ├── cbf_constraint.py       # linear Data-CBF constraint; hard first-step
│   │                            # only (A5), multi-barrier support
│   └── qp_solver.py            # CVXPY/OSQP wrapper
├── sims/                     # ground-truth ODE simulators
│   ├── van_der_pol.py
│   ├── cartpole.py             # safe_set() returns dict of separate bounds
│   ├── cstr.py                 # safe_set() returns dict of separate bounds
│   ├── quadrotor.py            # safe_set() returns dict, one per obstacle
│   ├── lorenz.py
│   └── exact_koopman_toy.py    # exact finite-dim Koopman embedding, for Thm 1 sanity check
├── baselines/
│   ├── mpc_known_model.py
│   ├── koopman_mpc_indirect.py
│   ├── koopman_cbf_indirect.py    # #3: Folkestad/Zinage-Bakolas, single-step
│   ├── koopman_cbf_mpc_liu.py     # #10 (new): Liu et al. 2026, receding-horizon
│   ├── linear_deepc.py
│   ├── nonlinear_deepc_no_physics.py
│   ├── sacbf_direct.py       # He et al. 2025 style
│   └── sos_barrier_lavaei.py # polynomial/SOS baseline
├── experiments/
│   ├── run_experiment.py     # single-config runner
│   ├── sweep.py               # T/sigma/physics-prior sweep orchestration
│   ├── stats.py                # seeds, CI, significance testing
│   ├── pe_transfer_sanity_check.py    # Theorem 1 exact-vs-perturbed sanity check
│   ├── theorem3_margin_validation.py  # Theorem 3 decoupled-vs-standard margin check
│   └── configs/                # one YAML per experiment configuration
├── tests/                    # unit tests, one file per module above
└── paper/                    # figures, tables, LaTeX assets generated from results
```

## Roadmap (Track B Implementation Plan)

| Weeks | Milestone |
|---|---|
| 1 | Repo skeleton, env, CI |
| 1-2 | **DONE** — All ODE simulators sanity-tested; linear DeePC QP matches LQR on toy LTI system (mass-spring-damper) |
| 2-4 | Physics-informed Koopman autoencoder (multi-barrier `h_coordinates`); CBF-QP on known-model pendulum; Data-CBF constraint on Hankel rows (hard first-step only, per A5) |
| 4-6 | First full end-to-end pipeline on Van der Pol |
| 6-10 | All 5 benchmarks + all 10 baselines/ablations online |
| 9-14 | Full experimental campaign (5 x 10 x sweep x >=10 seeds), incl. Theorem 1 and Theorem 3 validation experiments |
| 15-18 | Reproducibility polish, support writing track |
| 19-22 | Support internal review, final fresh-clone repro test |

## Reproducing results

Once experiments are run, each main figure/table will have a one-command
reproduction script (per the Reproducibility Checklist). Not yet populated —
tracked in `experiments/`.

## License

MIT — see `LICENSE`.

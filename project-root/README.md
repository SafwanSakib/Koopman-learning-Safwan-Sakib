# Physics-Informed, Safety-Certified Direct Data-Driven Predictive Control

Implementation codebase for *"A Physics-Informed Data-Enabled Predictive Control
Framework with Formal Safety Certificates for Unknown Nonlinear Systems."*

This is **Track B** of the project (see the project's Research Plan and
Track B Implementation Plan documents for full context). It implements:

- A physics-informed Koopman autoencoder lifting (`models/`)
- A DeePC (Data-Enabled Predictive Control) QP with a linear Data-CBF safety
  constraint expressed directly on Hankel-matrix rows (`control/`)
- Five nonlinear benchmark simulators (`sims/`)
- Nine baselines/ablations for fair comparison (`baselines/`)
- A statistically rigorous experiment runner (≥10–20 seeds, CIs, significance
  tests) (`experiments/`)

## Status

🚧 Repo skeleton — Week 1 deliverable. Modules are stubbed with docstrings
and TODOs matching the Track B plan's phase structure. See each module's
docstring for what it should do and which plan section it corresponds to.

## Setup

```bash
conda env create -f environment.yml
conda activate physics-informed-safe-deepc
pytest -q   # should discover and run tests (skeleton has smoke tests only)
```

## Repository structure

```
project-root/
├── environment.yml        # pinned exact versions
├── README.md
├── LICENSE                 # MIT
├── models/                 # physics-informed Koopman autoencoder
│   ├── autoencoder.py
│   ├── losses.py            # reconstruction, linear-dynamics, physics terms
│   └── train.py
├── control/                 # DeePC + CBF QP machinery
│   ├── deepc.py              # core Hankel-matrix QP
│   ├── cbf_constraint.py     # linear Data-CBF constraint construction
│   └── qp_solver.py          # CVXPY/OSQP wrapper
├── sims/                    # ground-truth ODE simulators
│   ├── van_der_pol.py
│   ├── cartpole.py
│   ├── cstr.py
│   ├── quadrotor.py
│   └── lorenz.py
├── baselines/
│   ├── mpc_known_model.py
│   ├── koopman_mpc_indirect.py
│   ├── koopman_cbf_indirect.py
│   ├── linear_deepc.py
│   ├── nonlinear_deepc_no_physics.py
│   ├── sacbf_direct.py       # He et al. 2025 style
│   └── sos_barrier_lavaei.py # polynomial/SOS baseline
├── experiments/
│   ├── run_experiment.py     # single-config runner
│   ├── sweep.py               # T/sigma/physics-prior sweep orchestration
│   ├── stats.py                # seeds, CI, significance testing
│   └── configs/                # one YAML per experiment configuration
├── tests/                    # unit tests, one file per module above
└── paper/                    # figures, tables, LaTeX assets generated from results
```

## Roadmap (Track B Implementation Plan)

| Weeks | Milestone |
|---|---|
| 1 | Repo skeleton, env, CI (this commit) |
| 1–2 | All 5 ODE simulators sanity-tested; linear DeePC QP matches LQR on toy LTI system |
| 2–4 | Physics-informed Koopman autoencoder; CBF-QP on known-model pendulum; Data-CBF constraint on Hankel rows |
| 4–6 | First full end-to-end pipeline on Van der Pol |
| 6–10 | All 5 benchmarks + all 9 baselines/ablations online |
| 9–14 | Full experimental campaign (5 × 9 × sweep × ≥10 seeds) |
| 15–18 | Reproducibility polish, support writing track |
| 19–22 | Support internal review, final fresh-clone repro test |

## Reproducing results

Once experiments are run, each main figure/table will have a one-command
reproduction script (per the Reproducibility Checklist). Not yet populated —
tracked in `experiments/`.

## License

MIT — see `LICENSE`.

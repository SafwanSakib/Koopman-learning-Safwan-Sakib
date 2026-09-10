"""Theorem 1 sanity-check experiment: PE-transfer, exact embedding -> perturbed
embedding.

Added 2026-09-05 per Track C's literature lock
(Consolidated_Findings_TrackA_TrackB_Revisions.md, Part 3.2, decision
confirmed by user 2026-09-05: "Yes, add it as a new experiment module").

What this experiment does, in three stages:

    1. EXACT-EMBEDDING LIMIT (epsilon_phys = 0): using
       sims/exact_koopman_toy.py's known exact finite-dimensional Koopman
       embedding z = exact_lift(x), generate a persistently-exciting raw
       trajectory, build the raw Hankel matrix H_L(u^d) and the lifted
       Hankel matrix H_L(z^d), and verify PE-transfer holds exactly: if the
       raw data is PE of order L+n, the lifted Hankel matrix is full rank.
       This reproduces (numerically, as a sanity check -- NOT a new proof)
       the case Shang, Cortes, Zheng (2024) prove analytically. If this
       stage fails, something is wrong with our Hankel-matrix machinery
       itself (controllers/deepc.py), not with the novel part of Theorem 1 --
       debug there first.

    2. PERTURBED EMBEDDING (epsilon_phys > 0): replace the exact lift with
       a perturbed version z_eps(x) = exact_lift(x) + eps * noise(x) (or,
       later, an actually-trained models.autoencoder.KoopmanAutoencoder
       standing in for the perturbation -- see TODO), sweep eps over a
       range, and track the smallest singular value of the lifted Hankel
       matrix (Theoretical_Background_Full.md Sec.10.1, Weyl's inequality is
       the expected proof tool) as a function of eps. This is the empirical
       curve Theorem 1's Weyl-perturbation argument should predict the
       shape of, once that argument exists (Track A).

    3. FIGURE: smallest singular value (or a full-rank boolean, or the
       PE-order margin) vs. eps, with the eps=0 point highlighted as the
       Shang et al. anchor. This is the "sanity check that visually
       demonstrates the theorem" figure referenced in the literature-lock
       memo -- low implementation cost, intended for the paper's Theorem 1
       section (Contribution 1, Introduction_Literature_Review_Draft.md
       Sec.1.4 item 1) as a complement to (not replacement for) the formal
       proof.

Relationship to controllers/deepc.py: this experiment is a *consumer* of
build_hankel_matrix and check_persistency_of_excitation -- implement those
first (Week 1-2 milestone), then this module becomes straightforward to
fill in. Until then, this module documents the exact experimental protocol
so Track A and Track B stay in sync on what "the sanity check" concretely
means.

TODO:
    - Implement once controllers/deepc.py::build_hankel_matrix and
      check_persistency_of_excitation are done (Week 1-2)
    - Stage 2's perturbation model: start with additive structured noise
      (cheap, isolates the rank-perturbation question cleanly); swap in an
      actual undertrained/small-data KoopmanAutoencoder as a second,
      more-realistic perturbation source once models/autoencoder.py is far
      enough along (Week 2-4+) -- comparing the two perturbation sources is
      itself a useful robustness check on the sanity-check figure.
    - Coordinate with Track A on whether the eps-sweep range should be
      chosen to match the specific bound constants that come out of the
      Weyl's-inequality proof attempt, once that exists, so the figure
      overlays the theoretical prediction line (same pattern as the
      Theorem 3 theory-vs-practice plot in experiments/run_experiment.py).
"""

from __future__ import annotations

import numpy as np


def perturb_lift(exact_z: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """Stage 2 perturbation: exact_z + epsilon * structured noise.

    This IS implemented now (not a stub) since it's simple and needed early
    for anyone iterating on the experiment protocol; the noise structure may
    be revisited once Track A's actual epsilon_phys definition (from the
    physics-informed training loss, models/losses.py) is finalized, to make
    sure this synthetic perturbation is representative of what a real
    undertrained autoencoder's error looks like.
    """
    return exact_z + epsilon * rng.standard_normal(exact_z.shape)


def run_exact_embedding_check(t_span: tuple[float, float] = (0.0, 100.0),
                               dt: float = 0.05, L: int = 10, n: int = 2,
                               seed: int = 0) -> dict:
    """Stage 1: verify PE-transfer holds exactly at epsilon_phys = 0 (the
    Shang et al. limit). Returns a dict with the raw-PE boolean, the
    lifted-Hankel-rank boolean, and the smallest singular value of each.
    """
    raise NotImplementedError(
        "Depends on controllers.deepc.build_hankel_matrix / "
        "check_persistency_of_excitation (Week 1-2). See module docstring, Stage 1."
    )


def run_perturbation_sweep(epsilons: np.ndarray, t_span: tuple[float, float] = (0.0, 100.0),
                            dt: float = 0.05, L: int = 10, n: int = 2,
                            seed: int = 0) -> dict:
    """Stage 2: sweep epsilon, return smallest singular value of the
    perturbed lifted Hankel matrix at each epsilon (plus the eps=0 anchor
    point from run_exact_embedding_check for the figure's reference line).
    """
    raise NotImplementedError(
        "Depends on controllers.deepc.build_hankel_matrix (Week 1-2). "
        "See module docstring, Stage 2."
    )


def make_sanity_check_figure(sweep_results: dict, out_path: str) -> None:
    """Stage 3: render the eps vs. smallest-singular-value figure, with the
    eps=0 (Shang et al. exact-case) point highlighted, and (once available)
    the Theorem 1 theoretical bound overlaid as a reference line.
    """
    raise NotImplementedError("Implement once Stage 1/2 produce real data")

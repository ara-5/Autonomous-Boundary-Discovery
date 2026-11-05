# A Benchmark for Black-Box Blind-Spot Audits of Detectors

**Status:** technical report on a research prototype. All numbers are produced by the code in this repository and saved verbatim in [`results/`](../results). Nothing here has been peer reviewed.

## 1. Summary

Given a detector that can only be queried, can an agent, within a fixed experiment budget, (a) learn *where* the detector fails, (b) identify a variable the detector secretly depends on, and (c) produce findings that hold on conditions it never explored?

I built four tasks with ground truth, a reference method (boundary-targeting experiment design), and several baselines. Main findings:

1. **The reference method is not a general win.** Against uniform sampling it helps on the two simulated grid tasks, adds nothing on a synthetic learned detector, and loses on real KDD Cup 99 data. Against a stronger space-filling baseline (maximin) the margin shrinks or disappears everywhere, and a published level-set method (straddle) ties or beats it on three of four tasks; my method keeps a clear edge only on withheld conditions of one grid task.
2. **Fair baselines matter.** Uniform fuzzing understates what a simple space-filling design achieves; the choice of baseline changed several conclusions.
3. **Targeted design for missing-variable discovery is the most promising result.** It beat random and space-filling design on the grid task and beat the space-filling baseline (not random) on a second, learned-detector task.
4. **The audit pipeline itself works.** It recovers 62% to 93% of a real detector's truly missed records (depending on strategy), returns readable rules, and a paired-intervention test exposes a variable the detector should not depend on.

The contribution I would stand behind is the **benchmark and the fair comparison**, not a claim that targeted exploration beats fuzzing.

## 2. Setup

**Tasks** (`python -m abd.benchmark --list`)

| Task | Detector | Ground truth of failure |
|---|---|---|
| `grid_delta` | IEEE 14-bus DC state estimator with residual bad-data test (AC power flow is reality) | perturbation large enough to matter but indistinguishable from none |
| `grid_adaptive` | same | perturbation below an alarm threshold operators raise with load |
| `ml_synthetic` | gradient-boosted classifier on synthetic traffic features, trained with a coverage gap | detector says "benign" on a real attack |
| `kdd_real` | gradient-boosted classifier on real KDD Cup 99 records; twelve rare attack families held out of training | detector says "benign" on a real attack record |

**Agent and metric.** The agent chooses experiments in a 3-dimensional condition space. A readable rule (decision tree) and a logistic rule with interaction features are fit to its observations. Score = balanced accuracy of the rule against ground truth, on (a) conditions the agent could explore and (b) **withheld** conditions (the highest values of one axis), which tests extrapolation. Mean ± 95% CI over 20 seeds, 60 experiments per run unless stated.

**Strategies.** `uniform`, `random` (with a safe-expansion radius), `uncertainty` sampling, `discriminating` (committee disagreement), `boundary` (disagreement plus classifier uncertainty near the estimated boundary), `hybrid` (half uniform, then boundary), and `maximin` (always probe the point farthest from everything tried). Custom strategies plug in with `register_strategy`.

## 3. Results

### 3.1 Blind-spot mapping (logistic rule, explored / withheld)

| Task | Boundary | Uniform | Maximin |
|---|---|---|---|
| `grid_delta` | **0.993 ± 0.002** / **0.965 ± 0.014** | 0.885 / 0.829 | 0.964 ± 0.012 / 0.927 ± 0.041 |
| `grid_adaptive` | 0.976 ± 0.009 / **0.975 ± 0.007** | 0.875 / 0.797 | 0.949 ± 0.025 / 0.895 ± 0.055 |
| `ml_synthetic` | 0.864 ± 0.011 / 0.788 ± 0.050 | 0.856 / 0.692 | 0.872 ± 0.010 / 0.807 ± 0.075 |
| `kdd_real` | 0.783 ± 0.017 / 0.686 ± 0.026 | 0.865 / 0.709 | 0.854 ± 0.009 / **0.774 ± 0.023** (no radius) |

Boundary-targeting beats uniform and maximin on explored conditions of `grid_delta` and on withheld conditions of `grid_adaptive`; elsewhere intervals overlap or it loses.

**Published-method baselines** (same protocol; [raw output](../results/benchmark_literature_baselines_results.txt)). `straddle` is active level-set estimation (Bryan et al. 2005; Gotovos et al. 2013) and `bo_falsify` is Bayesian-optimisation falsification, both on a Gaussian-process model of the distance to the blind-spot threshold:

| Task | Straddle | BO falsification | Boundary (mine) |
|---|---|---|---|
| `grid_delta` | 0.990 ± 0.008 / 0.963 ± 0.017 | 0.940 ± 0.025 / 0.927 ± 0.050 | 0.993 ± 0.002 / 0.965 ± 0.014 |
| `grid_adaptive` | 0.982 ± 0.007 / 0.881 ± 0.061 | 0.915 ± 0.025 / 0.954 ± 0.009 | 0.976 ± 0.009 / **0.975 ± 0.007** |
| `ml_synthetic` | 0.869 ± 0.016 / 0.758 ± 0.062 | 0.753 ± 0.040 / 0.643 ± 0.066 | 0.864 ± 0.011 / 0.788 ± 0.050 |
| `kdd_real` (radius 0.2) | **0.847 ± 0.009** / 0.672 ± 0.022 | 0.800 ± 0.024 / 0.629 ± 0.032 | 0.783 ± 0.017 / 0.686 ± 0.026 |

Straddle ties or beats my reference method on three of four tasks and is clearly better on explored conditions of the real data. My method keeps one clear advantage, withheld conditions of `grid_adaptive` (0.975 vs 0.881 for straddle and 0.954 for BO falsification). BO falsification, which searches for failures rather than mapping the boundary, is the weakest. The decision-tree rule does not transfer to withheld conditions when the boundary moves with the withheld axis (0.825 on `grid_adaptive`), whereas the logistic rule with interaction features does.

### 3.2 Why the real-data loss is not a simple artefact

On `kdd_real`, three follow-up tests failed to rescue boundary-targeting:

| Test | Boundary-targeting (explored / withheld) | Reference |
|---|---|---|
| Remove the safe-expansion radius | 0.760 / 0.681 | uniform 0.865 / 0.709 |
| Coverage-first hybrid | 0.807 / 0.702 | uniform 0.865 / 0.709 |
| 200 experiments instead of 60 | 0.772 / 0.718 | uniform 0.868 / 0.728 |

So the loss is not explained by the radius, by insufficient coverage, or by a small budget. A plausible but untested explanation is that real blind spots here are scattered clusters of rare families that a coarse three-feature projection captures poorly.

### 3.3 Missing-variable discovery

A hidden driver changes the detector's score; the agent reads one noisy proxy and two distractor channels and must name the real one. Success = only the real channel is flagged.

| Real driver found (of 20) | Grid, hidden temperature | Learned detector, drift 4.0 | Learned detector, drift 2.5 |
|---|---|---|---|
| random | 12 | 16 | 4 |
| **targeted** | **18** | **19** | 5 |
| maximin | 9 | 13 | 4 |
| random + confirm | 11 | 6 | 1 |
| targeted + confirm | **18** | 11 | 1 |
| maximin + confirm | 8 | 4 | 0 |

Grid: targeted + confirm vs maximin + confirm, 18/20 vs 8/20, Fisher p = 0.002; vs random + confirm, p = 0.031. Learned detector at drift 4.0: targeted vs maximin, p = 0.044 (without confirmation) and 0.048 (with); vs random, p = 0.34 and 0.20 (not significant). At drift 2.5 every method is near the floor. The confirmation step removes false identifications but costs power. The two tasks use different protocols (see 5).

### 3.4 A shortcut test on a learned detector

Holding a strong visible attack fixed and changing only a context variable that should be irrelevant lowers the detector's score on attacks by 0.73 (Wilcoxon p = 1.6e-11, n = 60), versus 0.12 on benign traffic (also significant, about six times smaller). The training data had attacks only at low context, and the planted gap is recovered as a measurable dependence.

## 4. Limitations

- **Three of four tasks are simulators I built**, and the planted driver, the training gap and the three-input parameterisation are my construction. They show what works in those sandboxes.
- **One real dataset**, and an old one: KDD Cup 99 has known redundancy and artefacts, and the held-out families are a design choice. A newer dataset is the most important missing check.
- **Modest statistics**: 20 seeds per cell, several overlapping intervals, a thin blind region on the grid (about 4% of impactful perturbations).
- **Definitions are modelling choices**: what counts as "blind", the withheld axis, and the three exposed features.
- **Missing-variable evidence is two planted drivers.** Effect size matters: at weak drift nothing works.
- **Published-method baselines are limited to two** (level-set estimation and BO falsification) in simple Gaussian-process implementations of my own, not the authors' code; safe-exploration methods with formal guarantees (SafeOpt-style) are not compared.

## 5. Protocol differences to be aware of

- The grid missing-variable task uses a quadratic baseline, a quadratic-interaction sensitivity estimate, and a fixed high-load confirmation batch. The learned-detector task uses a random-forest baseline with out-of-bag residuals, a random-forest sensitivity estimate, and confirmation that follows each strategy's own design policy. My first targeted design and confirmation step failed on the learned detector because they assumed the grid's structure; I rebuilt them once and ran the full comparison planned in advance, with both drift strengths fixed before running.
- The `boundary` rows on `kdd_real` use a 0.2 safe-expansion radius; the maximin figure in 3.1 is the no-radius run. With the same radius maximin scores 0.845 / 0.681.

## 6. Reproducing

```bash
pip install -r requirements.txt
python -m pytest -q
python -m abd.benchmark --task kdd_real --strategies uniform maximin boundary --seeds 20
python run_missing.py 20 --env ml --drift 4.0
```

Verbatim outputs for every table above are in [`results/`](../results). Prior art and related work are listed in the main [README](../README.md#limitations-and-prior-art).

## 7. What would make this convincing

1. A newer real dataset (for example CICIoT2023) and several model families.
2. Authors' own implementations of the literature baselines, plus safe-exploration methods with formal guarantees.
3. Missing-variable discovery on systems I did not design.
4. Explicit hypothesis-based experiment selection (expected information gain) instead of committee disagreement.

## 8. Responsible use

Defensive and simulation-only. The perturbations are synthetic measurement shifts or synthetic feature vectors used to find where a detector's model is unreliable so it can be hardened. The repository contains no tooling for real systems and none should be added.

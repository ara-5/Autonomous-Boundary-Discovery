"""Blind-spot map: agent vs fuzzing baselines, with a withheld-load transfer test."""
import sys

import numpy as np
from scipy import stats

from abd.blindspot import DELTA, IMPACT, LOAD_CAP, BlindAgent, adaptive_threshold
from abd.grid_env import GridDetectorEnv

T = np.random.default_rng(99).uniform(0, 1, (2500, 3))
T = T[T[:, 1] > IMPACT]                                        # only perturbations that matter
_e = GridDetectorEnv()
_null = _e.truth(np.c_[T[:, 0], np.zeros(len(T)), T[:, 2]])
TRUE = {"delta": (_e.truth(T) - _null) < DELTA, "adaptive": _e.truth(T) < adaptive_threshold(_null)}
SEEN = T[:, 0] <= LOAD_CAP


def bal_acc(pred, true):
    if not true.any() or true.all():
        return np.nan
    return 0.5 * (pred[true].mean() + (~pred[~true]).mean())


def trial(strategy, seed, mode, budget=60):
    env = GridDetectorEnv(noise_seed=seed)
    a = BlindAgent(env, strategy, step_radius=0.2, seed=seed, label_mode=mode)
    a.bootstrap(n=25, hi=0.6)
    a.calibrate_null()
    a.explore(budget)
    a.blind_rule()
    tb, out = TRUE[mode], []
    for p in (a.predict_blind(T), a.predict_blind_logit(T)):
        out += [bal_acc(p[SEEN], tb[SEEN]), bal_acc(p[~SEEN], tb[~SEEN])]
    return (*out, a)


def ci(v):
    v = np.array(v, float); v = v[~np.isnan(v)]
    h = stats.sem(v) * stats.t.ppf(0.975, len(v) - 1)
    return f"{v.mean():.3f} ± {h:.3f}"


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for mode in ["delta", "adaptive"]:
        tb = TRUE[mode]
        print(f"\n##### blind-spot type: {mode}  (true share: seen loads {tb[SEEN].mean():.2f}, withheld {tb[~SEEN].mean():.2f})")
        for s in ["uniform", "random", "uncertainty", "discriminating", "boundary"]:
            R = [trial(s, i, mode) for i in range(n)]
            print(f"{s:15s} TREE seen {ci([r[0] for r in R])} withheld {ci([r[1] for r in R])} | "
                  f"LOGIT seen {ci([r[2] for r in R])} withheld {ci([r[3] for r in R])}", flush=True)
            if s == "boundary":
                ex = R[0][4]
        print("Example tree rule (boundary agent, seed 0):\n" + ex.rules())

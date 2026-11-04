"""AI-security: audit a LEARNED detector. Blind-spot map with withheld contexts + shortcut test.

Prints in the same format as run_blind.py so scripts/make_results.py can chart it.
"""
import sys

import numpy as np
from scipy import stats

from abd.blindspot import IMPACT, LOAD_CAP, BlindAgent
from abd.ml_detector_env import MLDetectorEnv
from run_blind import bal_acc, ci

_e = MLDetectorEnv()
T = np.random.default_rng(99).uniform(0, 1, (6000, 3))
T = T[T[:, 1] > IMPACT]
TRUE = _e.truth(T) < 0.5
SEEN = T[:, 0] <= LOAD_CAP


def trial(strategy, seed, budget=60):
    env = MLDetectorEnv(noise_seed=seed)
    a = BlindAgent(env, strategy, step_radius=0.2, seed=seed, label_mode="absolute")
    a.bootstrap(n=25, hi=0.6)
    a.calibrate_null()
    a.explore(budget)
    a.blind_rule()
    out = []
    for p in (a.predict_blind(T), a.predict_blind_logit(T)):
        out += [bal_acc(p[SEEN], TRUE[SEEN]), bal_acc(p[~SEEN], TRUE[~SEEN])]
    return (*out, a)


def shortcut_test(n=60, seed=0):
    """Paired intervention: same attack, only the context knob changes. Context should be irrelevant."""
    env = MLDetectorEnv(noise_seed=seed)
    rng = np.random.default_rng(seed)
    base = np.c_[np.zeros(n), rng.uniform(0.7, 1, n), rng.uniform(0.7, 1, n)]   # strong, visible attacks
    lo, hi = base.copy(), base.copy()
    lo[:, 0], hi[:, 0] = rng.uniform(0.1, 0.4, n), rng.uniform(0.85, 1.0, n)
    d_att = np.array([env.run(h) - env.run(l) for l, h in zip(lo, hi)])
    ben = base.copy(); ben[:, 1] = 0
    blo, bhi = ben.copy(), ben.copy(); blo[:, 0], bhi[:, 0] = lo[:, 0], hi[:, 0]
    d_ben = np.array([env.run(h) - env.run(l) for l, h in zip(blo, bhi)])
    return d_att, d_ben


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"##### blind-spot type: ml_detector  (true share: seen contexts {TRUE[SEEN].mean():.2f}, withheld {TRUE[~SEEN].mean():.2f})")
    ex = None
    for s in ["uniform", "random", "uncertainty", "discriminating", "boundary"]:
        R = [trial(s, i) for i in range(n)]
        print(f"{s:15s} TREE seen {ci([r[0] for r in R])} withheld {ci([r[1] for r in R])} | "
              f"LOGIT seen {ci([r[2] for r in R])} withheld {ci([r[3] for r in R])}", flush=True)
        if s == "boundary":
            ex = R[0][4]
    print("Example tree rule (boundary agent, seed 0):\n" + ex.rules())
    da, db = shortcut_test()
    print("\n##### shortcut test (paired intervention: strong visible attack, context 0.1-0.4 -> 0.85-1.0)")
    print(f"mean change in detector score on attacks : {da.mean():+.3f}  (Wilcoxon p = {stats.wilcoxon(da)[1]:.2g}, n={len(da)})")
    print(f"mean change in detector score on benign  : {db.mean():+.3f}  (Wilcoxon p = {stats.wilcoxon(db)[1]:.2g}, n={len(db)})")

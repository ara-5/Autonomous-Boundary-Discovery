"""Real-data audit (KDD Cup 99): where does a learned intrusion detector go blind, found black-box?

Same output format as run_blind.py so scripts/make_results.py can chart it.
"""
import sys

import numpy as np

from abd.blindspot import LOAD_CAP, BlindAgent
from abd.real_detector_env import UNSEEN, RealDetectorEnv
from run_blind import bal_acc, ci

_e = RealDetectorEnv()
T = np.random.default_rng(99).uniform(0, 1, (6000, 3))
TRUE = _e.truth(T) < 0.5
SEEN = T[:, 0] <= LOAD_CAP
POOL_MISS = _e.p < 0.5
FAM = _e.pool["labels"].values


def trial(strategy, seed, budget=60):
    env = RealDetectorEnv(noise_seed=seed)
    a = BlindAgent(env, strategy, step_radius=0.2, seed=seed, label_mode="real")
    a.bootstrap(n=25, hi=0.6)
    a.calibrate_null()
    a.explore(budget)
    a.blind_rule()
    out = []
    for p in (a.predict_blind(T), a.predict_blind_logit(T)):
        out += [bal_acc(p[SEEN], TRUE[SEEN]), bal_acc(p[~SEEN], TRUE[~SEEN])]
    return (*out, a, a.predict_blind_logit(env.ranks))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rep = _e.detector_report()
    print("##### ground truth: detector recall by attack family (probe pool, never shown to the agent)")
    print("held-out families (not in training): " + ", ".join(UNSEEN))
    print("lowest recall : " + ", ".join(f"{k} {v:.2f}" for k, v in rep.head(8).items()))
    print(f"pool of real attack records: {len(_e.pool)}, missed by the detector: {POOL_MISS.mean():.1%}")
    print(f"\n##### blind-spot type: real_kdd  (true share: seen contexts {TRUE[SEEN].mean():.2f}, withheld {TRUE[~SEEN].mean():.2f})")
    store, ex = {}, None
    for s in ["uniform", "random", "uncertainty", "discriminating", "boundary"]:
        R = [trial(s, i) for i in range(n)]
        store[s] = np.array([r[5] for r in R])
        print(f"{s:15s} TREE seen {ci([r[0] for r in R])} withheld {ci([r[1] for r in R])} | "
              f"LOGIT seen {ci([r[2] for r in R])} withheld {ci([r[3] for r in R])}", flush=True)
        if s == "boundary":
            ex = R[0][4]
    print("\n##### share of REAL records the agent's rule flags as a blind spot (mean over seeds, logistic rule)")
    print(f"{'':16s}{'missed families':>16s}{'caught families':>17s}")
    for s, P in store.items():
        flag = P.mean(0)                                   # per-record flag frequency across seeds
        print(f"{s:16s}{flag[POOL_MISS].mean():16.2f}{flag[~POOL_MISS].mean():17.2f}")
    print("(first column = fraction of truly-missed attack records flagged; second = false-flag rate on detected ones)")
    print("\nExample tree rule (boundary agent, seed 0; axes: load=dst_bytes rank, magnitude=src_bytes rank, non_stealth=count rank):\n" + ex.rules())

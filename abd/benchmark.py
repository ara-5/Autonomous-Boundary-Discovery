"""Blind-spot audit benchmark.

A task is a black-box detector plus ground truth for where it fails. A strategy decides
which experiments to run within a fixed budget. The score is how well the blind-spot rule
learned from those experiments predicts the detector's true failures on (a) conditions the
agent explored and (b) conditions that were WITHHELD from exploration.

    python -m abd.benchmark --list
    python -m abd.benchmark --task grid_delta --strategies uniform maximin boundary --seeds 10

Add your own strategy:
    from abd.blindspot import register_strategy
    register_strategy("mine", lambda X, y, C, rng, labels: int(rng.integers(len(C))))
"""
import argparse
import json

import numpy as np
from scipy import stats

from .blindspot import (CUSTOM_STRATEGIES, DELTA, IMPACT, LOAD_CAP, BlindAgent,
                        adaptive_threshold)

BUILTIN = ["uniform", "random", "uncertainty", "discriminating", "boundary", "hybrid"]
_CACHE = {}


def bal_acc(pred, true):
    if not true.any() or true.all():
        return np.nan
    return 0.5 * (pred[true].mean() + (~pred[~true]).mean())


class Task:
    def __init__(self, name, description, make_env, mode, T, true):
        self.name, self.description, self.make_env, self.mode = name, description, make_env, mode
        self.T, self.true, self.seen = T, true, T[:, 0] <= LOAD_CAP


def _grid(mode):
    from .grid_env import GridDetectorEnv
    e = GridDetectorEnv()
    T = np.random.default_rng(99).uniform(0, 1, (2500, 3))
    T = T[T[:, 1] > IMPACT]
    null = e.truth(np.c_[T[:, 0], np.zeros(len(T)), T[:, 2]])
    true = (e.truth(T) - null) < DELTA if mode == "delta" else e.truth(T) < adaptive_threshold(null)
    desc = {"delta": "IEEE-14 grid detector; blind = indistinguishable from no perturbation",
            "adaptive": "IEEE-14 grid detector; blind = below the load-adaptive alarm threshold"}[mode]
    return Task(f"grid_{mode}", desc, lambda seed: GridDetectorEnv(noise_seed=seed), mode, T, true)


def _ml():
    from .ml_detector_env import MLDetectorEnv
    e = MLDetectorEnv()
    T = np.random.default_rng(99).uniform(0, 1, (6000, 3))
    T = T[T[:, 1] > IMPACT]
    return Task("ml_synthetic", "learned detector on synthetic traffic with a planted training gap",
                lambda seed: MLDetectorEnv(noise_seed=seed), "absolute", T, e.truth(T) < 0.5)


def _kdd():
    from .real_detector_env import RealDetectorEnv
    e = RealDetectorEnv()
    T = np.random.default_rng(99).uniform(0, 1, (6000, 3))
    return Task("kdd_real", "learned detector on real KDD Cup 99 records (held-out attack families)",
                lambda seed: RealDetectorEnv(noise_seed=seed), "real", T, e.truth(T) < 0.5)


TASKS = {"grid_delta": lambda: _grid("delta"), "grid_adaptive": lambda: _grid("adaptive"),
         "ml_synthetic": _ml, "kdd_real": _kdd}


def get_task(name):
    if name not in _CACHE:
        _CACHE[name] = TASKS[name]()
    return _CACHE[name]


def run_trial(task, strategy, seed, budget=60, radius=0.2):
    env = task.make_env(seed)
    a = BlindAgent(env, strategy, step_radius=radius, seed=seed, label_mode=task.mode)
    a.bootstrap(n=25, hi=0.6)
    a.calibrate_null()
    a.explore(budget)
    a.blind_rule()
    out = {}
    for rule, pred in (("tree", a.predict_blind(task.T)), ("logit", a.predict_blind_logit(task.T))):
        out[f"{rule}_seen"] = bal_acc(pred[task.seen], task.true[task.seen])
        out[f"{rule}_withheld"] = bal_acc(pred[~task.seen], task.true[~task.seen])
    return out


def ci(v):
    v = np.array(v, float)
    v = v[~np.isnan(v)]
    h = stats.sem(v) * stats.t.ppf(0.975, len(v) - 1) if len(v) > 1 else float("nan")
    return float(v.mean()), float(h)


def run_task(task_name, strategies, seeds=10, budget=60, radius=0.2, verbose=True):
    task = get_task(task_name)
    results = {}
    if verbose:
        print(f"##### blind-spot type: {task.name}  (true share: seen contexts {task.true[task.seen].mean():.2f}, "
              f"withheld {task.true[~task.seen].mean():.2f})   [{task.description}]")
    for s in strategies:
        trials = [run_trial(task, s, i, budget, radius) for i in range(seeds)]
        results[s] = {k: ci([t[k] for t in trials]) for k in trials[0]}
        if verbose:
            r = results[s]
            f = lambda k: f"{r[k][0]:.3f} ± {r[k][1]:.3f}"
            print(f"{s:15s} TREE seen {f('tree_seen')} withheld {f('tree_withheld')} | "
                  f"LOGIT seen {f('logit_seen')} withheld {f('logit_withheld')}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list tasks and strategies")
    ap.add_argument("--task", default="grid_delta", choices=list(TASKS))
    ap.add_argument("--strategies", nargs="+", default=["uniform", "maximin", "boundary"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--radius", default="0.2", help="safe-expansion radius, or 'none'")
    ap.add_argument("--json", help="write results to this JSON file")
    a = ap.parse_args()
    if a.list:
        print("tasks:", ", ".join(TASKS)); print("strategies:", ", ".join(BUILTIN + sorted(CUSTOM_STRATEGIES)))
        return
    radius = None if a.radius.lower() == "none" else float(a.radius)
    res = run_task(a.task, a.strategies, a.seeds, a.budget, radius)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

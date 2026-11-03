"""Where does the grid's bad-data detector stop being reliable? Agent vs random, same budget."""
import numpy as np

from abd.agent import Agent
from abd.grid_env import GridDetectorEnv

TEST = np.random.default_rng(99).uniform(0, 1, (1500, 3))


def trial(strategy, seed, truth_test, budget=60):
    env = GridDetectorEnv(noise_seed=seed)
    a = Agent(env, strategy, step_radius=0.2, seed=seed)
    a.bootstrap(n=25, hi=0.6)
    a.explore(budget)
    n = a.failure_boundary(kind="tree")
    a.update()
    fail = np.abs(a.baseline.predict(TEST) - truth_test) > 3 * a.sigma  # where baseline is really wrong
    out = dict(found=a.clf is not None, n_bad=a.n_bad,
               rmse_base=float(np.sqrt(np.mean((a.baseline.predict(TEST) - truth_test) ** 2))),
               rmse_upd=float(np.sqrt(np.mean((a.updated.predict(TEST) - truth_test) ** 2))))
    if a.clf is not None and fail.any() and (~fail).any():
        pred = a.clf.predict(TEST).astype(bool)
        out["bal_acc"] = 0.5 * (pred[fail].mean() + (~pred[~fail]).mean())
        out["w"] = np.round(n, 2)
    return out


if __name__ == "__main__":
    truth_test = GridDetectorEnv().truth(TEST)
    for s in ["random", "discriminating", "boundary"]:
        R = [trial(s, i, truth_test) for i in range(8)]
        g = lambda k: np.nanmean([r.get(k, np.nan) for r in R])
        print(f"\n== {s} == boundary found {sum(r['found'] for r in R)}/{len(R)}, failures seen {g('n_bad'):.1f}")
        print(f"  held-out RMSE baseline {g('rmse_base'):.2f} -> updated {g('rmse_upd'):.2f}")
        print(f"  boundary balanced acc (vs where baseline truly fails): {g('bal_acc'):.3f}")
        ws = np.mean([np.abs(r["w"]) for r in R if "w" in r], axis=0)
        print(f"  mean tree feature importance over (load, magnitude, non-stealth): {np.round(ws, 2)}")

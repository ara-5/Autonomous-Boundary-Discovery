import numpy as np

from abd.agent import Agent
from abd.env import HiddenRegimeEnv


def trial(strategy, seed, budget=60):
    env = HiddenRegimeEnv(seed=seed)
    a = Agent(env, strategy, seed=seed)
    a.bootstrap()
    a.explore(budget)
    n = a.failure_boundary()
    a.update()
    # withheld conditions: fresh uniform samples over the whole space, truth known
    T = np.random.default_rng(10_000 + seed).uniform(0, 1, (4000, 3))
    y = env.truth(T)

    def rmse(m):
        return float(np.sqrt(np.mean((m.predict(T) - y) ** 2)))

    out = dict(rmse_base=rmse(a.baseline), rmse_upd=rmse(a.updated),
               viol=env.violations, found=a.clf is not None, n_bad=a.n_bad)
    if a.clf is not None:
        pred, true = a.clf.predict(T).astype(bool), env.in_hidden_regime(T)
        out["bal_acc"] = 0.5 * (pred[true].mean() + (~pred[~true]).mean())
        out["angle"] = float(np.degrees(np.arccos(min(1, abs(n @ np.array([1, 1, 0]) / np.sqrt(2))))))
        out["x2_weight"] = float(abs(n[2]))
    return out


if __name__ == "__main__":
    for s in ["random", "discriminating"]:
        R = [trial(s, i) for i in range(10)]

        def g(k):
            v = [r[k] for r in R if k in r]
            return np.mean(v) if v else float("nan")

        print(f"\n== {s} ==  boundary found in {sum(r['found'] for r in R)}/{len(R)} runs")
        print(f"  failures seen in sandbox : {g('n_bad'):.1f}   safety violations: {g('viol'):.2f}")
        print(f"  held-out RMSE baseline {g('rmse_base'):.3f} -> updated {g('rmse_upd'):.3f}")
        print(f"  boundary balanced acc    : {g('bal_acc'):.3f}")
        print(f"  boundary angle error     : {g('angle'):.1f} deg (true normal = x0+x1); x2 weight {g('x2_weight'):.2f} (should be ~0)")

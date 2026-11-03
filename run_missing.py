"""Missing-variable discovery: does the agent notice that its model lacks a driver,
identify WHICH context channel it is, and expose it with fewer experiments?

Strategies for choosing experiment conditions x:
  random   : x uniform
  targeted : after a pilot, pick x where the fitted context-interaction effect is largest
Identification rule: a channel is 'discovered' when the partial-correlation of the
x-only model residual with that channel is significant (Bonferroni, p<0.01) AND it is
the only such channel. Cost = experiments needed.
"""
import sys

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from abd.grid_ctx_env import CHANNELS, GridContextEnv

BATCH, PILOT, MAXN, CONFIRM_N = 10, 20, 200, 30


def px(X):
    return PolynomialFeatures(2).fit_transform(X)


def interaction_model(X, C, y):
    """y ~ poly2(x) + sum_j c_j * [1, x]"""
    F = [px(X)] + [C[:, [j]] * np.c_[np.ones(len(X)), X] for j in range(C.shape[1])]
    return Ridge(alpha=1e-2).fit(np.hstack(F), y), F


def test_channels(X, C, y):
    res = y - make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(alpha=1e-2)).fit(X, y).predict(X)
    p = [stats.pearsonr(C[:, j], res)[1] for j in range(C.shape[1])]
    return np.array(p)


def trial(strategy, seed, confirm=False):
    env = GridContextEnv(noise_seed=seed) if not hasattr(trial, "env") else trial.env
    trial.env = env
    env.rng = np.random.default_rng(seed)
    rng = np.random.default_rng(1000 + seed)
    X, C, y = [], [], []

    def go(x):
        y.append(env.run(x)); X.append(x); C.append(env.last_ctx)

    for x in rng.uniform(0, 1, (PILOT, 3)):
        go(x)
    while len(y) < MAXN:
        p = test_channels(np.array(X), np.array(C), np.array(y))
        sig = p < 0.01 / 3
        if sig.sum() == 1:
            j = int(np.argmax(sig))
            if not confirm:
                return len(y), CHANNELS[j], True
            # Confirmation: fresh targeted batch, test ONLY channel j on out-of-sample residuals.
            Xa, ya = np.array(X), np.array(y)
            base = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(alpha=1e-2)).fit(Xa, ya)
            Xf, Cf, yf = [], [], []
            cand = rng.uniform(0, 1, (800, 3))
            for x in cand[np.argsort(np.abs(cand[:, 0] - 1))[:CONFIRM_N]]:   # high-load, high-sensitivity region
                yf.append(env.run(x)); Xf.append(x); Cf.append(env.last_ctx)
            r, pv = stats.pearsonr(np.array(Cf)[:, j], np.array(yf) - base.predict(np.array(Xf)))
            X += Xf; C += Cf; y += yf
            if pv < 0.01:
                return len(y), CHANNELS[j], True
            continue                                   # rejected: keep exploring
        if strategy == "random":
            nxt = rng.uniform(0, 1, (BATCH, 3))
        else:
            cand = rng.uniform(0, 1, (800, 3))
            m, _ = interaction_model(np.array(X), np.array(C), np.array(y))
            nj = len(CHANNELS)
            w = m.coef_[px(cand[:1]).shape[1]:].reshape(nj, 4)         # per-channel [1, x] weights
            eff = np.abs(np.c_[np.ones(len(cand)), cand] @ w.T).sum(1)  # sensitivity to context
            nxt = cand[np.argsort(-eff)[:BATCH]]
        for x in nxt:
            go(x)
    return MAXN, None, False


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for strat, conf in [("random", False), ("targeted", False), ("random", True), ("targeted", True)]:
        R = [trial(strat, i, conf) for i in range(n)]
        cost = np.array([r[0] for r in R])
        ok = np.array([r[2] and r[1] == "ambient_sensor" for r in R])
        wrong = np.array([r[2] and r[1] != "ambient_sensor" for r in R])
        tag = strat + ("+confirm" if conf else "")
        print(f"{tag:17s} correct {ok.sum()}/{n}, wrong channel {wrong.sum()}; "
              f"experiments: median {np.median(cost):.0f}, mean {cost.mean():.0f} (cap {MAXN})", flush=True)

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

_ENV = sys.argv[sys.argv.index("--env") + 1] if "--env" in sys.argv else "grid"
if _ENV == "bldg":
    from abd.bldg_ctx_env import CHANNELS, BldgContextEnv as ContextEnv   # real building data (LBNL Bldg 59), hidden occupancy
elif _ENV == "ml":
    from abd.ml_ctx_env import CHANNELS, MLContextEnv as ContextEnv   # learned detector, hidden traffic mix
else:
    from abd.grid_ctx_env import CHANNELS, GridContextEnv as ContextEnv   # grid detector, hidden temperature
TRUE_CHANNEL = CHANNELS[0]
USE_RF = _ENV in ("ml", "bldg")   # S-shaped detector response: use a flexible x-only model
if USE_RF and "--drift" in sys.argv:
    import abd.ml_ctx_env as _mlc
    _mlc.DRIFT = float(sys.argv[sys.argv.index("--drift") + 1])
from sklearn.ensemble import RandomForestRegressor

BATCH, PILOT, MAXN, CONFIRM_N = 10, 20, 200, 30


def px(X):
    return PolynomialFeatures(2).fit_transform(X)


def interaction_model(X, C, y):
    """y ~ poly2(x) + sum_j c_j * [1, x]"""
    F = [px(X)] + [C[:, [j]] * np.c_[np.ones(len(X)), X] for j in range(C.shape[1])]
    return Ridge(alpha=1e-2).fit(np.hstack(F), y), F


def x_only_model(X, y):
    """The agent's current model: no context inputs. Returns (model, residuals on the training data)."""
    if USE_RF:
        rf = RandomForestRegressor(150, min_samples_leaf=3, oob_score=True, random_state=0).fit(X, y)
        return rf, y - rf.oob_prediction_            # out-of-bag residuals, so no in-sample optimism
    m = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(alpha=1e-2)).fit(X, y)
    return m, y - m.predict(X)


def test_channels(X, C, y):
    res = x_only_model(X, y)[1]
    p = [stats.pearsonr(C[:, j], res)[1] for j in range(C.shape[1])]
    return np.array(p)


def sensitivity_rf(X, C, y, cand):
    """Flexible-model sensitivity: how much does the prediction move when each context channel is swung
    from 0.1 to 0.9, at each candidate experiment? (used for the learned-detector task)"""
    rf = RandomForestRegressor(150, min_samples_leaf=3, random_state=0).fit(np.c_[X, C], y)
    base = np.c_[cand, np.tile(C.mean(0), (len(cand), 1))]
    eff = np.zeros(len(cand))
    for j in range(C.shape[1]):
        lo, hi = base.copy(), base.copy()
        lo[:, 3 + j], hi[:, 3 + j] = 0.1, 0.9
        eff += np.abs(rf.predict(hi) - rf.predict(lo))
    return eff


def choose(strategy, X, C, y, rng, n):
    """Pick the next n experiment conditions under a design strategy."""
    if strategy == "random":
        return rng.uniform(0, 1, (n, 3))
    cand = rng.uniform(0, 1, (800, 3))
    if strategy == "maximin":                            # space-filling design: farthest from everything so far
        chosen, pts = [], np.array(X)
        for _ in range(n):
            d = np.min(np.linalg.norm(cand[:, None] - pts[None], axis=2), axis=1)
            k = int(np.argmax(d)); chosen.append(cand[k]); pts = np.vstack([pts, cand[k]])
        return np.array(chosen)
    if USE_RF:
        eff = sensitivity_rf(np.array(X), np.array(C), np.array(y), cand)
    else:
        m, _ = interaction_model(np.array(X), np.array(C), np.array(y))
        w = m.coef_[px(cand[:1]).shape[1]:].reshape(len(CHANNELS), 4)    # per-channel [1, x] weights
        eff = np.abs(np.c_[np.ones(len(cand)), cand] @ w.T).sum(1)        # sensitivity to context
    return cand[np.argsort(-eff)[:n]]


def trial(strategy, seed, confirm=False):
    env = ContextEnv(noise_seed=seed) if not hasattr(trial, "env") else trial.env
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
            base, _ = x_only_model(Xa, ya)
            Xf, Cf, yf = [], [], []
            if USE_RF:      # learned-detector task: confirm with the strategy's own design policy
                batch = choose(strategy, X, C, y, rng, CONFIRM_N)
            else:           # grid task (legacy protocol, kept so saved results stay valid): high-load region
                cand = rng.uniform(0, 1, (800, 3))
                batch = cand[np.argsort(np.abs(cand[:, 0] - 1))[:CONFIRM_N]]
            for x in batch:
                yf.append(env.run(x)); Xf.append(x); Cf.append(env.last_ctx)
            r, pv = stats.pearsonr(np.array(Cf)[:, j], np.array(yf) - base.predict(np.array(Xf)))
            X += Xf; C += Cf; y += yf
            if pv < 0.01:
                return len(y), CHANNELS[j], True
            continue                                   # rejected: keep exploring
        for x in choose(strategy, X, C, y, rng, BATCH):
            go(x)
    return MAXN, None, False


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    runs = [("maximin", False), ("maximin", True)] if "--maximin" in sys.argv else [("random", False), ("targeted", False), ("random", True), ("targeted", True)]
    for strat, conf in runs:
        R = [trial(strat, i, conf) for i in range(n)]
        cost = np.array([r[0] for r in R])
        ok = np.array([r[2] and r[1] == TRUE_CHANNEL for r in R])
        wrong = np.array([r[2] and r[1] != TRUE_CHANNEL for r in R])
        tag = strat + ("+confirm" if conf else "")
        print(f"{tag:17s} correct {ok.sum()}/{n}, wrong channel {wrong.sum()}; "
              f"experiments: median {np.median(cost):.0f}, mean {cost.mean():.0f} (cap {MAXN})", flush=True)

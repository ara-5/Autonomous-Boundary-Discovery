"""AI-security testbed: audit a LEARNED detector (synthetic tabular traffic, no real attacks).

A gradient-boosted classifier is trained to separate benign from attack feature vectors.
Its training data has a coverage gap, like real security datasets: attacks were only
recorded in low-context conditions and with clearly visible signatures. The detector can
therefore learn a *shortcut* (context) and has never seen quiet attacks or attacks in
high-context conditions.

Agent knobs, x in [0,1]^3 (same layout as the grid sandbox so one engine serves both):
  x0 : context (think network load / time of day). Never part of an attack signature.
  x1 : attack intensity (ground truth: attack present iff x1 > 0.3)
  x2 : visibility of the attack signature (0 = attack perfectly mimics benign traffic)
Output: detector's attack probability on one noisy synthetic sample.
Everything is synthetic feature vectors; nothing here models or generates real attacks.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

DIM = 6
NOISE = 0.35
TRAIN_CTX_MAX = 0.5      # attacks were only ever recorded at context <= this
TRAIN_VIS_MIN = 0.5      # ... and only with clearly visible signatures
CTX_SHIFT = 2.5          # how strongly context moves ordinary traffic


class MLDetectorEnv:
    dim = 3
    safety_limit = 1e9

    def __init__(self, noise_seed=0, model_seed=0):
        rng = np.random.default_rng(777)             # fixed world: same traffic physics every run
        self.d = rng.normal(size=DIM); self.d *= 3.5 / np.linalg.norm(self.d)
        self.m0, self.m1 = rng.normal(size=DIM), rng.normal(size=DIM) * CTX_SHIFT
        self.rng = np.random.default_rng(noise_seed)
        self.violations = 0
        self.n_queries = 0
        self.model = self._train(np.random.default_rng(model_seed))

    def _mean(self, X):
        X = np.atleast_2d(X)
        return self.m0 + X[:, [0]] * self.m1 + (X[:, [1]] * X[:, [2]]) * self.d

    def _train(self, rng, n=6000):
        k = rng.uniform(0, 1, n)
        is_att = rng.random(n) < 0.35
        k = np.where(is_att, rng.uniform(0, TRAIN_CTX_MAX, n), k)          # attacks only at low context
        inten = np.where(is_att, rng.uniform(0.35, 1, n), 0.0)
        vis = np.where(is_att, rng.uniform(TRAIN_VIS_MIN, 1, n), rng.uniform(0, 1, n))
        F = self._mean(np.c_[k, inten, vis]) + rng.normal(0, NOISE, (n, DIM))
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0).fit(F, is_att)

    # ground truth, used only for scoring
    def truth(self, X):
        """Expected detector probability on noise-free features."""
        return self.model.predict_proba(self._mean(X))[:, 1]

    def run(self, x):
        self.n_queries += 1
        f = self._mean(np.asarray(x, float)) + self.rng.normal(0, NOISE, (1, DIM))
        return float(self.model.predict_proba(f)[0, 1])

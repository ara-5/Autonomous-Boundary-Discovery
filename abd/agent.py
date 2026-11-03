"""Boundary-discovery agent: committee world model -> discriminating experiments
-> failure boundary -> updated model verified on withheld conditions."""
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")


def _poly(d):
    return make_pipeline(PolynomialFeatures(d), StandardScaler(), Ridge(alpha=1e-3))


def make_committee(seed=0):
    """Competing explanations of the data. They agree where data is dense and
    diverge where the data under-determines the model -> a knowledge boundary."""
    return [
        _poly(1), _poly(2), _poly(3),
        RandomForestRegressor(50, min_samples_leaf=2, random_state=seed),
        GaussianProcessRegressor(RBF(0.5) + WhiteKernel(0.01), normalize_y=True),
    ]


class Agent:
    def __init__(self, env, strategy="discriminating", step_radius=0.2,
                 safety_margin=1.0, seed=0):
        self.env, self.strategy = env, strategy
        self.r, self.margin = step_radius, safety_margin
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.X, self.y = [], []
        self.clf = None

    # 1. initial world model from "historical data" in a narrow safe region
    def bootstrap(self, n=25, hi=0.6):
        for x in self.rng.uniform(0, hi, (n, self.env.dim)):
            self._observe(x)
        Xa, ya = self._arrays()
        self.baseline = _poly(2).fit(Xa, ya)  # the "current understanding"
        res = ya - self.baseline.predict(Xa)
        self.sigma = max(res.std(), 0.05)

    def _observe(self, x):
        self.X.append(x)
        self.y.append(self.env.run(x))

    def _arrays(self):
        return np.array(self.X), np.array(self.y)

    # 2. knowledge boundary = where competing models disagree
    def disagreement(self, C, committee):
        P = np.stack([m.predict(C) for m in committee])
        return P.std(0), P

    # 3+4. choose a safe experiment that best discriminates between models
    def step(self, n_cand=1500):
        Xa, ya = self._arrays()
        C = self.rng.uniform(0, 1, (n_cand, self.env.dim))
        dist = np.min(np.linalg.norm(C[:, None] - Xa[None], axis=2), axis=1)
        C = C[dist <= self.r]  # safe-expansion radius around known points
        if self.strategy == "random":
            self._observe(C[self.rng.integers(len(C))])
            return
        committee = [m.fit(Xa, ya) for m in make_committee(self.seed)]
        score, P = self.disagreement(C, committee)
        if self.strategy == "boundary":
            # also seek points near the *estimated failure boundary* (classifier uncertainty)
            bad = np.abs(ya - self.baseline.predict(Xa)) > 3 * self.sigma
            if bad.sum() >= 3 and (~bad).sum() >= 3:
                rf = RandomForestClassifier(100, min_samples_leaf=2, random_state=self.seed).fit(Xa, bad)
                unc = 1 - np.abs(2 * rf.predict_proba(C)[:, 1] - 1)
                score = score / (score.max() + 1e-9) + unc
        ok = P.max(0) + self.margin < self.env.safety_limit  # conservative safety filter
        if ok.any():
            score = np.where(ok, score, -np.inf)
        self._observe(C[np.argmax(score)])

    def explore(self, budget=60):
        for _ in range(budget):
            self.step()

    # 5. explain where the baseline stops predicting reality
    def failure_boundary(self, k=3.0, kind="logistic"):
        Xa, ya = self._arrays()
        bad = np.abs(ya - self.baseline.predict(Xa)) > k * self.sigma
        self.n_bad = int(bad.sum())
        if bad.sum() < 3 or (~bad).sum() < 3:
            self.clf = None
            return None
        if kind == "tree":  # nonlinear, human-readable rules
            self.clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3,
                                              class_weight="balanced", random_state=0).fit(Xa, bad)
            return self.clf.feature_importances_
        self.clf = make_pipeline(StandardScaler(with_mean=False),
                                 LogisticRegression(C=100.0, class_weight="balanced")).fit(Xa, bad)
        w = self.clf[-1].coef_[0] / self.clf[0].scale_  # back to input units
        return w / np.linalg.norm(w)  # boundary normal direction

    def rules(self, names):
        return export_text(self.clf, feature_names=list(names)) if self.clf is not None else ""

    # 6. update the model; caller verifies it on withheld conditions
    def update(self):
        Xa, ya = self._arrays()
        if self.clf is None:
            self.updated = self.baseline
            return
        gate = self.clf.predict(Xa).astype(bool)
        res = ya - self.baseline.predict(Xa)
        corr = _poly(2).fit(Xa[gate], res[gate]) if gate.sum() > 6 else None
        base, clf = self.baseline, self.clf

        class Updated:
            @staticmethod
            def predict(X):
                p = base.predict(X)
                if corr is None:
                    return p
                g = clf.predict(X).astype(bool)
                if g.any():
                    p[g] += corr.predict(X[g])
                return p

        self.updated = Updated

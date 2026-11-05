"""Blind-spot discovery for the grid detector.

Blind spot := the perturbation is large enough to matter (x1 > IMPACT) but the detector
score is indistinguishable (< DELTA) from its own no-perturbation score at that load.
The agent learns the 'no-perturbation' reference itself with a few calibration runs, so
it never needs ground truth. Exploration is restricted to loads <= LOAD_CAP; higher loads
are withheld and used only to test whether the discovered rule transfers.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from .agent import Agent, _poly, make_committee

IMPACT, DELTA, LOAD_CAP = 0.4, 3.0, 0.7
FLAG_MIN, FLAG_MULT = 6.2, 1.5   # adaptive mode: operators raise the alarm threshold with load


# Plug-in strategies: name -> fn(X, y, candidates, rng, labels) -> index into candidates.
#   X, y        : experiments so far (inputs in [0,1]^3, detector outputs)
#   candidates  : (n, 3) array of allowed next experiments (safety radius already applied)
#   labels(X,y) : returns (impact_mask, blind_mask) using the task's blind-spot definition
CUSTOM_STRATEGIES = {}


def register_strategy(name, fn):
    CUSTOM_STRATEGIES[name] = fn


def _maximin(X, y, C, rng, labels):
    """Space-filling baseline: pick the candidate farthest from everything tried so far."""
    return int(np.argmax(np.min(np.linalg.norm(C[:, None] - X[None], axis=2), axis=1)))


register_strategy("maximin", _maximin)

# Agent-aware strategies: name -> fn(agent, X, y, candidates) -> index. They may use agent.blind_margin(X, y),
# the continuous distance to the blind-spot threshold (blind <=> impactful and blind_margin < 0).
AGENT_STRATEGIES = {}


def register_agent_strategy(name, fn):
    AGENT_STRATEGIES[name] = fn


def _gp_on_margin(agent, X, y):
    """GP regression of the blind-spot margin on the impactful experiments so far (None if too few)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    keep = agent._impact(X)
    if keep.sum() < 8:
        return None
    gp = GaussianProcessRegressor(RBF([0.3, 0.3, 0.3]) + WhiteKernel(0.05), normalize_y=True, n_restarts_optimizer=0)
    return gp.fit(X[keep], agent.blind_margin(X, y)[keep])


def _straddle(agent, X, y, C):
    """Active level-set estimation (Bryan et al. 2005; Gotovos et al. 2013): probe where the
    margin's confidence band straddles the threshold: argmax 1.96*sigma - |mu|."""
    gp = _gp_on_margin(agent, X, y)
    if gp is None:
        return int(agent.rng.integers(len(C)))
    mu, sd = gp.predict(C, return_std=True)
    score = np.where(agent._impact(C), 1.96 * sd - np.abs(mu), -np.inf)
    return int(np.argmax(score))


def _bo_falsify(agent, X, y, C):
    """Bayesian-optimisation falsification: search for the most failing input by minimising the
    lower confidence bound of the margin (mu - 1.96*sigma)."""
    gp = _gp_on_margin(agent, X, y)
    if gp is None:
        return int(agent.rng.integers(len(C)))
    mu, sd = gp.predict(C, return_std=True)
    score = np.where(agent._impact(C), -(mu - 1.96 * sd), -np.inf)
    return int(np.argmax(score))


register_agent_strategy("straddle", _straddle)
register_agent_strategy("bo_falsify", _bo_falsify)


def adaptive_threshold(null_score):
    return np.maximum(FLAG_MIN, FLAG_MULT * null_score)


def _feats(X):
    X = np.atleast_2d(X)
    return np.c_[X, X[:, 0] * X[:, 2], X[:, 1] * X[:, 2], X[:, 0] * X[:, 1], np.log(X[:, 2] + 0.02)]


class BlindAgent(Agent):
    def __init__(self, *a, label_mode="delta", **k):
        super().__init__(*a, **k)
        self.mode = label_mode

    def calibrate_null(self, n=10):
        L = self.rng.uniform(0, LOAD_CAP, n)
        z = [self.env.run([l, 0.0, self.rng.uniform()]) for l in L]
        self.null = _poly(2).fit(L[:, None], z)

    def _impact(self, X):
        # real-data mode: every probed record is a genuine attack, so every point is 'impactful'
        return np.ones(len(X), bool) if self.mode == "real" else X[:, 1] > IMPACT

    def blind_margin(self, X, y):
        """Continuous distance to the blind-spot threshold: blind <=> impactful and margin < 0."""
        ref = self.null.predict(X[:, :1])
        if self.mode in ("absolute", "real"):
            return y - 0.5
        if self.mode == "adaptive":
            return y - adaptive_threshold(ref)
        return y - ref - DELTA

    def labels(self, X, y):
        impact = self._impact(X)
        ref = self.null.predict(X[:, :1])
        if self.mode in ("absolute", "real"):
            blind = impact & (y < 0.5)                       # learned detector says 'benign' on a real attack
        elif self.mode == "adaptive":
            blind = impact & (y < adaptive_threshold(ref))   # detector would not raise an alarm
        else:
            blind = impact & ((y - ref) < DELTA)             # indistinguishable from no perturbation
        return impact, blind

    HYBRID_FRAC = 0.5    # 'hybrid': uniform coverage for this share of the budget, then boundary-targeting

    def explore(self, budget=60):
        self._budget, self._t = budget, 0
        for _ in range(budget):
            self.step()
            self._t += 1

    def step(self, n_cand=1500):
        strat = self.strategy
        if strat == "hybrid":
            strat = "uniform" if self._t < self.HYBRID_FRAC * self._budget else "boundary"
        Xa, ya = self._arrays()
        C = self.rng.uniform(0, 1, (n_cand, 3))
        C[:, 0] *= LOAD_CAP                                   # high loads are withheld
        if strat != "uniform" and self.r is not None:   # r=None: no safe-expansion radius
            d = np.min(np.linalg.norm(C[:, None] - Xa[None], axis=2), axis=1)
            C = C[d <= self.r]
        if strat in AGENT_STRATEGIES:
            self._observe(C[AGENT_STRATEGIES[strat](self, Xa, ya, C)]); return
        if strat in CUSTOM_STRATEGIES:
            self._observe(C[CUSTOM_STRATEGIES[strat](Xa, ya, C, self.rng, self.labels)]); return
        if strat in ("uniform", "random"):
            self._observe(C[self.rng.integers(len(C))]); return
        committee = [m.fit(Xa, ya) for m in make_committee(self.seed)]
        score, _ = self.disagreement(C, committee)
        if strat in ("boundary", "uncertainty"):
            impact, blind = self.labels(Xa, ya)
            if blind.sum() >= 2 and (impact & ~blind).sum() >= 2:
                rf = RandomForestClassifier(100, min_samples_leaf=2, random_state=self.seed)
                rf.fit(Xa[impact], blind[impact])
                unc = 1 - np.abs(2 * rf.predict_proba(C)[:, 1] - 1)
                unc = np.where(self._impact(C), unc, 0)       # only boundary inside impact zone
                score = unc if strat == "uncertainty" else score / (score.max() + 1e-9) + unc
            elif strat == "uncertainty":
                score = self.rng.random(len(C))
        self._observe(C[np.argmax(score)])

    def blind_rule(self):
        Xa, ya = self._arrays()
        impact, blind = self.labels(Xa, ya)
        if blind.sum() < 2 or (impact & ~blind).sum() < 2:
            self.tree = self.logit = None
            return
        self.logit = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, class_weight="balanced", max_iter=2000)
                                   ).fit(_feats(Xa[impact]), blind[impact])
        self.tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, class_weight="balanced",
                                           random_state=0).fit(Xa[impact], blind[impact])

    def rules(self):
        names = ["load", "magnitude", "non_stealth"]
        return export_text(self.tree, feature_names=names) if self.tree is not None else "(no rule found)"

    def predict_blind(self, X):
        return np.zeros(len(X), bool) if self.tree is None else self.tree.predict(X).astype(bool)

    def predict_blind_logit(self, X):
        return np.zeros(len(X), bool) if self.logit is None else self.logit.predict(_feats(X)).astype(bool)

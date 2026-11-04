"""Second missing-variable testbed: a LEARNED detector with a hidden driver it never saw.

Same synthetic traffic and gradient-boosted detector as ml_detector_env, plus a hidden
traffic-mix variable h drawn fresh every query. h shifts the feature vector along a fixed
direction; the detector was trained without it, and the agent's model has no input for it.
The agent can read three context channels, only the first of which is real:
    traffic_mix_sensor = h + noise (the real proxy)
    time_noise, clock  = distractors independent of h
Everything is synthetic feature vectors; nothing here models real attacks.
"""
import numpy as np

from .ml_detector_env import DIM, NOISE, MLDetectorEnv

CHANNELS = ["traffic_mix_sensor", "time_noise", "clock"]
DRIFT = 1.4


class MLContextEnv(MLDetectorEnv):
    def __init__(self, noise_seed=0, model_seed=0):
        super().__init__(noise_seed=noise_seed, model_seed=model_seed)
        g = np.random.default_rng(4242).normal(size=DIM)
        self.g = DRIFT * g / np.linalg.norm(g)
        self.last_ctx = None

    def run(self, x):
        self.n_queries += 1
        h = self.rng.uniform()
        self.last_ctx = np.array([h + self.rng.normal(0, 0.05), self.rng.uniform(), self.rng.uniform()])
        self.last_h = h
        f = self._mean(np.asarray(x, float)) + h * self.g + self.rng.normal(0, NOISE, (1, DIM))
        return float(self.model.predict_proba(f)[0, 1])

    def truth_h(self, X, H):
        X = np.atleast_2d(X)
        return self.model.predict_proba(self._mean(X) + np.asarray(H)[:, None] * self.g)[:, 1]

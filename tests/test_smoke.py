import numpy as np

from abd.agent import Agent
from abd.env import HiddenRegimeEnv
from abd.grid_env import GridDetectorEnv


def test_toy_env_hidden_regime():
    env = HiddenRegimeEnv()
    inside, outside = np.array([[0.9, 0.9, 0.0]]), np.array([[0.2, 0.2, 0.0]])
    assert env.in_hidden_regime(inside)[0] and not env.in_hidden_regime(outside)[0]
    assert env.truth(inside)[0] > 2 * env.truth(outside)[0]


def test_agent_finds_toy_boundary():
    env = HiddenRegimeEnv(seed=1)
    a = Agent(env, "discriminating", seed=1)
    a.bootstrap()
    a.explore(30)
    a.failure_boundary()
    assert a.n_bad > 0 and env.violations == 0


def test_physics_consistent_perturbation_is_invisible():
    """A perturbation inside the estimator's column space cannot raise the detector score."""
    env = GridDetectorEnv()
    clean = env.truth(np.array([[0.5, 0.0, 0.0]]))[0]
    stealth = env.truth(np.array([[0.5, 1.0, 0.0]]))[0]
    loud = env.truth(np.array([[0.5, 1.0, 1.0]]))[0]
    assert abs(stealth - clean) < 1e-6 and loud > 10 * clean

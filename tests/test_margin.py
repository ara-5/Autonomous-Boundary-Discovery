import numpy as np
import pytest

from abd.benchmark import get_task
from abd.blindspot import AGENT_STRATEGIES, BlindAgent


@pytest.mark.parametrize("task_name", ["grid_delta", "grid_adaptive", "ml_synthetic"])
def test_margin_agrees_with_binary_blind_labels(task_name):
    task = get_task(task_name)
    a = BlindAgent(task.make_env(0), "uniform", seed=0, label_mode=task.mode)
    a.bootstrap(n=25, hi=0.6)
    a.calibrate_null()
    X, y = np.array(a.X), np.array(a.y)
    impact, blind = a.labels(X, y)
    assert np.array_equal(blind, impact & (a.blind_margin(X, y) < 0))


def test_literature_strategies_registered_and_run():
    assert {"straddle", "bo_falsify"} <= set(AGENT_STRATEGIES)
    task = get_task("ml_synthetic")
    for name in ("straddle", "bo_falsify"):
        a = BlindAgent(task.make_env(1), name, step_radius=0.2, seed=1, label_mode=task.mode)
        a.bootstrap(n=25, hi=0.6)
        a.calibrate_null()
        a.explore(12)
        assert len(a.X) == 25 + 12     # calibration runs go straight to the env, not into the history

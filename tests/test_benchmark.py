import numpy as np

from abd.benchmark import get_task, run_trial
from abd.blindspot import CUSTOM_STRATEGIES, register_strategy


def test_custom_strategy_plugs_in_and_scores():
    calls = []

    def first_candidate(X, y, C, rng, labels):
        calls.append(len(X))
        return 0

    register_strategy("first", first_candidate)
    assert "first" in CUSTOM_STRATEGIES and "maximin" in CUSTOM_STRATEGIES
    task = get_task("grid_delta")
    out = run_trial(task, "first", seed=0, budget=8, radius=0.2)
    assert len(calls) == 8
    assert set(out) == {"tree_seen", "tree_withheld", "logit_seen", "logit_withheld"}
    assert all(np.isnan(v) or 0.0 <= v <= 1.0 for v in out.values())

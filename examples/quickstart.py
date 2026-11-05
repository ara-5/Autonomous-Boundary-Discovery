"""Quickstart: audit a detector with your own experiment-selection strategy and compare it to baselines.

    pip install -e ".[grid]"
    python examples/quickstart.py

Runs on the simulated IEEE-14 grid task (no data download). Scores are balanced accuracy of the learned
blind-spot rule on explored and on withheld conditions; see the README for how to read them.
"""
from abd.benchmark import run_task
from abd.blindspot import register_strategy


def farthest_from_last(X, y, candidates, rng, labels):
    """Toy strategy: probe the candidate farthest from the most recent experiment."""
    import numpy as np
    d = np.linalg.norm(candidates - np.asarray(X)[-1], axis=1)
    return int(np.argmax(d))


register_strategy("farthest_from_last", farthest_from_last)

if __name__ == "__main__":
    run_task("grid_delta", ["uniform", "maximin", "straddle", "farthest_from_last"], seeds=5)

import numpy as np

from abd.ml_ctx_env import CHANNELS, MLContextEnv


def test_hidden_driver_moves_detector_output_and_proxy_tracks_it():
    env = MLContextEnv(noise_seed=0)
    X = np.random.default_rng(0).uniform(0, 1, (500, 3))
    effect = np.abs(env.truth_h(X, np.ones(len(X))) - env.truth_h(X, np.zeros(len(X))))
    assert effect.max() > 0.5                      # the hidden driver matters somewhere
    ctx, h = [], []
    for x in X[:100]:
        env.run(x)
        ctx.append(env.last_ctx); h.append(env.last_h)
    ctx = np.array(ctx)
    assert len(CHANNELS) == ctx.shape[1] == 3
    assert np.corrcoef(ctx[:, 0], h)[0, 1] > 0.95          # real proxy tracks the hidden variable
    assert abs(np.corrcoef(ctx[:, 1], h)[0, 1]) < 0.4      # distractor does not

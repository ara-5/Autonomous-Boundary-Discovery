"""Real-data audit: a learned intrusion detector on the public KDD Cup 99 (10%) dataset.

Nothing is planted here. The detector is a gradient-boosted classifier trained on normal
traffic plus the common attack families; the rare remote-to-local / user-to-root families
are held out of training, which gives the detector a natural coverage gap.

The agent probes the detector with REAL attack records. A query is a point x in [0,1]^3,
the rank-normalised values of three interpretable features:
  x0 = destination bytes   (the 'context' axis; high ranks are withheld from exploration)
  x1 = source bytes
  x2 = connection count
The environment snaps x to the nearest real attack record and returns the detector's
attack probability for it (random choice among the 3 nearest, which gives query noise).
Ground truth used for scoring is the detector's decision on the nearest record.

Data note: KDD Cup 99 is old and has known redundancy/artifacts. It is used here as a
public, real, labelled benchmark for the method, not as a model of modern traffic.
Requires network on first use (scikit-learn downloads and caches the dataset).
"""
import functools

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.datasets import fetch_kddcup99
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

UNSEEN = ["warezclient", "guess_passwd", "buffer_overflow", "warezmaster", "imap", "rootkit",
          "loadmodule", "ftp_write", "multihop", "phf", "perl", "spy"]
KNOBS = ["dst_bytes", "src_bytes", "count"]


@functools.lru_cache(maxsize=1)
def _build(data_home=None):
    df = fetch_kddcup99(percent10=True, as_frame=True, data_home=data_home).frame
    df["labels"] = df["labels"].astype(str).str.strip(".")
    for c in ["protocol_type", "service", "flag"]:
        df[c] = df[c].astype(str).astype("category").cat.codes
    feat = [c for c in df.columns if c != "labels"]
    for c in feat:
        df[c] = pd.to_numeric(df[c])
    df["att"] = (df["labels"] != "normal").astype(int)
    feat = [c for c in df.columns if c not in ("labels", "att")]
    train = df[~df["labels"].isin(UNSEEN)]
    tr, te = train_test_split(train, test_size=0.3, random_state=0, stratify=train["labels"])
    model = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(tr[feat], tr["att"])
    pool = pd.concat([te[te["att"] == 1], df[df["labels"].isin(UNSEEN)]])
    pool = pool.groupby("labels", group_keys=False).sample(frac=1, random_state=0)   # shuffle, keep all cols
    pool = pool.groupby("labels", group_keys=False).head(1500).sample(frac=1, random_state=1).reset_index(drop=True)
    pool["p"] = model.predict_proba(pool[feat])[:, 1]
    ranks = np.column_stack([pool[k].rank(method="first").values / len(pool) for k in KNOBS])
    return pool, ranks, cKDTree(ranks)


class RealDetectorEnv:
    dim = 3
    safety_limit = 1e9

    def __init__(self, noise_seed=0, data_home=None):
        self.pool, self.ranks, self.tree = _build(data_home)
        self.p = self.pool["p"].values
        self.rng = np.random.default_rng(noise_seed)
        self.violations = 0
        self.n_queries = 0

    def truth(self, X):
        _, idx = self.tree.query(np.atleast_2d(X))
        return self.p[idx]

    def family_of(self, X):
        _, idx = self.tree.query(np.atleast_2d(X))
        return self.pool["labels"].values[idx]

    def run(self, x):
        self.n_queries += 1
        _, idx = self.tree.query(np.asarray(x, float), k=3)
        return float(self.p[self.rng.choice(idx)])

    def detector_report(self):
        g = self.pool.groupby("labels")["p"].agg(lambda s: float((s >= 0.5).mean()))
        return g.sort_values()

"""Real-data audit #2: a learned detector on UNSW-NB15 (2015 flows), a newer dataset than KDD Cup 99.

Same protocol as real_detector_env, with choices fixed BEFORE running:
  - four rare attack categories are held out of training: Analysis, Backdoor, Shellcode, Worms
  - knobs: x0 = dbytes rank (context axis, high ranks withheld), x1 = sbytes rank, x2 = ct_srv_src rank
  - pool = test-split attacks of seen categories + all held-out attacks (<=1500 per category)
  - detector = HistGradientBoosting; ground truth = detector decision (<0.5) on the nearest record

Data: not in the repo. Put the Train/Test CSVs of UNSW-NB15 (e.g. the Hugging Face mirror
Mireu-Lab/UNSW-NB15, GPL-3.0) in data/unsw/ as train.csv and test.csv. Academic use per UNSW.
"""
import functools
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from .real_detector_env import RealDetectorEnv

UNSEEN = ["Analysis", "Backdoor", "Shellcode", "Worms"]
KNOBS = ["dbytes", "sbytes", "ct_srv_src"]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "unsw")


@functools.lru_cache(maxsize=1)
def _build(data_dir=DATA_DIR):
    df = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in ("train.csv", "test.csv")],
                   ignore_index=True).drop(columns=["id"]).drop_duplicates().reset_index(drop=True)
    df["labels"] = df["attack_cat"].astype(str).str.strip()
    for c in ["proto", "service", "state"]:
        df[c] = df[c].astype(str).astype("category").cat.codes
    feat = [c for c in df.columns if c not in ("attack_cat", "label", "labels")]
    train = df[~df["labels"].isin(UNSEEN)]
    tr, te = train_test_split(train, test_size=0.3, random_state=0, stratify=train["labels"])
    model = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(tr[feat], tr["label"])
    pool = pd.concat([te[te["label"] == 1], df[df["labels"].isin(UNSEEN)]])
    pool = pool.groupby("labels", group_keys=False).sample(frac=1, random_state=0)
    pool = pool.groupby("labels", group_keys=False).head(1500).sample(frac=1, random_state=1).reset_index(drop=True)
    pool["p"] = model.predict_proba(pool[feat])[:, 1]
    ranks = np.column_stack([pool[k].rank(method="first").values / len(pool) for k in KNOBS])
    return pool, ranks, cKDTree(ranks)


class UNSWDetectorEnv(RealDetectorEnv):
    def __init__(self, noise_seed=0, data_dir=DATA_DIR):
        self.pool, self.ranks, self.tree = _build(data_dir)
        self.p = self.pool["p"].values
        self.rng = np.random.default_rng(noise_seed)
        self.violations = 0
        self.n_queries = 0

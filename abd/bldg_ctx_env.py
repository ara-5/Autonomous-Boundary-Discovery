"""Missing-variable discovery on a system I did not design: LBNL Building 59 (Berkeley office, 2018-2020).

Real hourly records. Target y = HVAC electric power (hvac_N + hvac_S). The agent's model knows three
conditions: x0 = outdoor air-temperature rank, x1 = solar-radiation rank, x2 = hour of day. It does not
know the day type or the occupant count. Three context channels are readable alongside each reading:
  occupancy              (camera occupant count, third + fourth floor south)   <- the candidate real driver
  occupancy_other_weeks  (the same signal taken 28 days away: same diurnal pattern, no real link)
  clock                  (uniform noise)
A query snaps x to the nearest real hourly record (random choice among 3 nearest = query noise).

Protocol fixed before running: success = only `occupancy` flagged. Ground truth is not planted; occupancy is
used because a feasibility check on all records showed it improves a random-forest HVAC model (CV R^2 0.59 ->
0.63). Honest limits: queries replay history (not interventions) and the effect is small.

Data: LBNL Building 59, CC0, https://zenodo.org/records/5951008 ; unzip to data/bldg59/ (git-ignored).
"""
import functools
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CHANNELS = ["occupancy", "occupancy_other_weeks", "clock"]
CLEAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "bldg59", "Bldg59_clean data")
SHIFT_HOURS = 28 * 24


def _rank01(s):
    return s.rank(method="first").values / len(s)


@functools.lru_cache(maxsize=1)
def _build(clean=CLEAN):
    e = pd.read_csv(os.path.join(clean, "ele.csv"), usecols=range(6), parse_dates=["date"]).set_index("date")
    o = pd.read_csv(os.path.join(clean, "occ.csv"), parse_dates=["date"]).set_index("date")
    w = pd.read_csv(os.path.join(clean, "site_weather.csv"), parse_dates=["date"]).set_index("date")
    H = pd.concat([e.resample("h").mean(), o.resample("h").mean(), w.resample("h").mean()], axis=1)
    H["occ"] = H.occ_third_south + H.occ_fourth_south
    H["occ_other"] = H["occ"].shift(-SHIFT_HOURS)            # same signal, 28 days later (hourly grid)
    H["hvac"] = H.hvac_N + H.hvac_S
    H = H.dropna(subset=["hvac", "occ", "occ_other", "air_temp_set_1", "solar_radiation_set_1"]).copy()
    H["noise"] = np.random.default_rng(0).uniform(0, 1, len(H))
    X = np.c_[_rank01(H.air_temp_set_1), _rank01(H.solar_radiation_set_1), H.index.hour.values / 23.0]
    C = np.c_[_rank01(H.occ), _rank01(H.occ_other), H.noise.values]
    return X, C, H.hvac.values.astype(float), cKDTree(X)


class BldgContextEnv:
    dim = 3

    def __init__(self, noise_seed=0, clean=CLEAN):
        self.X, self.C, self.y, self.tree = _build(clean)
        self.rng = np.random.default_rng(noise_seed)
        self.last_ctx = self.C[0]

    def run(self, x):
        _, idx = self.tree.query(np.asarray(x, float), k=3)
        i = int(self.rng.choice(idx))
        self.last_ctx = self.C[i]
        return float(self.y[i])

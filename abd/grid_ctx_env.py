"""Grid detector sandbox with a HIDDEN driver the agent's model does not contain.

Hidden driver h (ambient temperature, drawn fresh every query) stretches overhead-line
reactance by up to REACT_GAIN. The defender's DC estimator uses nominal reactance, so its
residual drifts with h. The agent never sees h; it can read three context channels:
    ambient_sensor = h + noise   (the real, noisy proxy)
    grid_noise, clock            (distractors, independent of h)
"""
import numpy as np
import pandapower as pp
import pandapower.networks as nw
from scipy.interpolate import RegularGridInterpolator

from .grid_env import GridDetectorEnv

CHANNELS = ["ambient_sensor", "grid_noise", "clock"]
REACT_GAIN = 0.8
HOT_LINES = slice(0, 8)   # only part of the network heats up (uneven, like real weather)


class GridContextEnv(GridDetectorEnv):
    def __init__(self, noise_seed=0, grid_pts=10, h_pts=5):
        super().__init__(noise_seed=noise_seed, grid_pts=grid_pts)
        net = nw.case14()
        x0 = net.line.x_ohm_per_km.copy()
        base_p, base_q = net.load.p_mw.copy(), net.load.q_mvar.copy()
        pp.runpp(net)
        lk = net._pd2ppc_lookups["bus"]
        hs = np.linspace(0, 1, h_pts)
        Z = np.zeros((h_pts, len(self.s_grid), self.Z.shape[1]))
        for i, h in enumerate(hs):
            xs = x0.copy()
            xs.iloc[HOT_LINES] = x0.iloc[HOT_LINES] * (1 + REACT_GAIN * h)
            net.line.x_ohm_per_km = xs
            for j, s in enumerate(self.s_grid):
                net.load.p_mw, net.load.q_mvar = base_p * s, base_q * s
                pp.runpp(net)
                pf = net.res_line.p_from_mw.values.tolist() + net.res_trafo.p_hv_mw.values.tolist()
                inj = np.zeros(self.nb)
                for bus in net.bus.index:
                    inj[lk[bus]] = -net.res_bus.p_mw[bus]
                Z[i, j] = np.concatenate([np.array(pf) / net.sn_mva, inj / net.sn_mva])
        self._interp = RegularGridInterpolator((hs, self.s_grid), Z)
        self.last_ctx = None

    def _score(self, x, h, noise):
        s = 0.6 + 0.9 * x[0]
        m = 2.5 * x[1]
        a = m * ((1 - x[2]) * self.a_stealth + x[2] * self.a_loud)
        z = self._interp([[h, s]])[0] + a
        if noise:
            z = z + self.rng.normal(0, self.sigma, len(z))
        r = self.S @ z
        return float(np.sqrt(r @ self.W @ r))

    def truth_h(self, X, H):
        return np.array([self._score(x, h, False) for x, h in zip(np.atleast_2d(X), H)])

    def run(self, x):
        """Returns detector score; context channels available in self.last_ctx."""
        self.n_queries += 1
        h = self.rng.uniform()
        self.last_ctx = np.array([h + self.rng.normal(0, 0.05), self.rng.uniform(), self.rng.uniform()])
        self.last_h = h
        return self._score(np.asarray(x, float), h, True)

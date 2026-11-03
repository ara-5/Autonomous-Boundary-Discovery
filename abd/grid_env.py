"""Defensive cyber-physical sandbox: IEEE-14 grid + DC state estimator + bad-data detector.

Truth   : AC power flow (pandapower) generates the real measurements.
Defender: a linear DC state estimator with a chi-square-style residual test.
Probe   : the agent injects a *synthetic* measurement perturbation in simulation to
          learn where the detector's model of the grid stops matching reality.
          Nothing here touches a real system; it is a blind-spot finder for the defender.

Inputs  x in [0,1]^3
  x0 : system loading factor      -> s = 0.6 + 0.9*x0
  x1 : perturbation magnitude     -> m = 0.25*x1 (rad-equivalent)
  x2 : non-stealth fraction beta  (0 = physics-consistent, 1 = pure residual-space)
Output y = detector score sqrt(J) (bigger = more likely flagged).
"""
import numpy as np
import pandapower as pp
import pandapower.networks as nw


class GridDetectorEnv:
    dim = 3
    safety_limit = 1e9  # simulation only; nothing physical to protect
    sigma = 0.01

    def __init__(self, noise_seed=0, grid_pts=19):
        self.rng = np.random.default_rng(noise_seed)
        self.violations = 0
        self.n_queries = 0
        net = nw.case14()
        pp.runpp(net)
        ppc = net._ppc
        lk = net._pd2ppc_lookups["bus"]
        br = ppc["branch"]
        nb = ppc["bus"].shape[0]
        f, t = br[:, 0].astype(int), br[:, 1].astype(int)
        b = 1.0 / (br[:, 3] * np.where(br[:, 8] == 0, 1, br[:, 8]))
        self.nb, self.nl = nb, len(f)
        A = np.zeros((self.nl, nb))
        A[np.arange(self.nl), f] = 1
        A[np.arange(self.nl), t] = -1
        Hf = b[:, None] * A                      # line flow from angles
        Hi = A.T @ Hf                            # bus injections from angles
        H = np.vstack([Hf, Hi])
        self.H = H[:, 1:]                        # slack angle fixed
        self.W = np.eye(len(H)) / self.sigma ** 2
        G = self.H.T @ self.W @ self.H
        K = self.H @ np.linalg.solve(G, self.H.T @ self.W)
        self.S = np.eye(len(H)) - K              # residual operator
        # true measurements (AC) on a loading grid, interpolated at run time
        self.s_grid = np.linspace(0.6, 1.5, grid_pts)
        base_p, base_q = net.load.p_mw.copy(), net.load.q_mvar.copy()
        Z = []
        for s in self.s_grid:
            net.load.p_mw, net.load.q_mvar = base_p * s, base_q * s
            pp.runpp(net)
            pf = (net.res_line.p_from_mw.values.tolist() + net.res_trafo.p_hv_mw.values.tolist())
            inj = np.zeros(nb)
            for i, bus in enumerate(net.bus.index):
                inj[lk[bus]] = -(net.res_bus.p_mw[bus])
            # order branches as in ppc (lines then trafos)
            Z.append(np.concatenate([np.array(pf) / net.sn_mva, inj / net.sn_mva]))
        self.Z = np.array(Z)
        rng = np.random.default_rng(1234)
        c = np.zeros(self.H.shape[1])
        c[3:8] = rng.normal(size=5)              # fixed attacked region
        self.a_stealth = self.H @ c
        self.a_stealth /= np.linalg.norm(self.a_stealth)
        d = self.S @ rng.normal(size=len(H))     # a residual-space direction
        self.a_loud = d / np.linalg.norm(d)

    def _z(self, s):
        return np.array([np.interp(s, self.s_grid, self.Z[:, j]) for j in range(self.Z.shape[1])])

    def _attacked(self, x, noise):
        s = 0.6 + 0.9 * x[0]
        m = 0.25 * x[1] * 10                      # scaled to per-unit measurement shift
        beta = x[2]
        a = m * ((1 - beta) * self.a_stealth + beta * self.a_loud)
        z = self._z(s) + a
        if noise:
            z = z + self.rng.normal(0, self.sigma, len(z))
        return z

    def truth(self, X):
        X = np.atleast_2d(X)
        return np.array([np.sqrt(self._J(self.S @ self._attacked(x, False))) for x in X])

    def _J(self, r):
        return float(r @ self.W @ r)

    def run(self, x):
        self.n_queries += 1
        return float(np.sqrt(self._J(self.S @ self._attacked(np.asarray(x, float), True))))

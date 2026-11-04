"""Generate README media from the real code: response map + two GIFs.

    python scripts/make_media.py
Outputs go to assets/. Everything is computed from the sandbox, nothing is mocked.
"""
import os
import sys

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abd.blindspot import DELTA, IMPACT, LOAD_CAP, BlindAgent, adaptive_threshold  # noqa: E402
from abd.grid_ctx_env import CHANNELS, GridContextEnv  # noqa: E402
from abd.grid_env import GridDetectorEnv  # noqa: E402
from abd.ml_detector_env import TRAIN_CTX_MAX, TRAIN_VIS_MIN, MLDetectorEnv  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(OUT, exist_ok=True)

# validated reference palette (light mode)
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": ["Segoe UI", "DejaVu Sans"], "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def frame_to_array(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


# ---------------------------------------------------------------- 1. response map
def response_map():
    env = GridDetectorEnv()
    bx = np.linspace(0, 0.3, 120)   # non-stealth fraction (zoomed on the interesting sliver)
    by = np.linspace(0.05, 1, 120)  # perturbation magnitude
    BX, BY = np.meshgrid(bx, by)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    for ax, load in zip(axes, [0.2, 0.9]):
        P = np.c_[np.full(BX.size, load), BY.ravel(), BX.ravel()]
        score = env.truth(P).reshape(BX.shape)
        null = env.truth(np.c_[P[:, 0], np.zeros(len(P)), P[:, 2]]).reshape(BX.shape)
        thr = adaptive_threshold(null)
        im = ax.imshow(score, origin="lower", extent=[0, 0.3, 0.05, 1], aspect="auto",
                       cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("s", SEQ),
                       vmin=0, vmax=60)
        ax.contour(BX, BY, (score - null), levels=[DELTA], colors=[ORANGE], linewidths=2)
        ax.contour(BX, BY, score - thr, levels=[0], colors=[MAGENTA], linewidths=2, linestyles="--")
        ax.set_title(f"System load factor {0.6 + 0.9 * load:.2f}", color=INK, fontsize=12, loc="left")
        ax.set_xlabel("non-stealth fraction (0 = physics-consistent perturbation)")
        ax.grid(False)
    axes[0].set_ylabel("perturbation magnitude")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("detector score  √J  (higher = more likely flagged)")
    axes[1].text(0.298, 0.08, "orange solid: indistinguishable from no perturbation\nmagenta dashed: below the load-adaptive alarm threshold",
                 ha="right", va="bottom", fontsize=8.5, color=INK2,
                 bbox=dict(facecolor=SURF, edgecolor=GRID, boxstyle="round,pad=0.4"))
    fig.suptitle("Where the grid's bad-data detector goes blind", x=0.01, ha="left", fontsize=14, color=INK)
    fig.savefig(os.path.join(OUT, "detector_response_map.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 1b. learned-detector map
def ml_map():
    env = MLDetectorEnv()
    cx = np.linspace(0, 1, 140)
    vy = np.linspace(0, 1, 140)
    CX, VY = np.meshgrid(cx, vy)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    for ax, inten in zip(axes, [0.5, 1.0]):
        P = env.truth(np.c_[CX.ravel(), np.full(CX.size, inten), VY.ravel()]).reshape(CX.shape)
        im = ax.imshow(P, origin="lower", extent=[0, 1, 0, 1], aspect="auto", vmin=0, vmax=1,
                       cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("s", SEQ))
        ax.contour(CX, VY, P, levels=[0.5], colors=[ORANGE], linewidths=2.2)
        ax.add_patch(plt.Rectangle((0, TRAIN_VIS_MIN), TRAIN_CTX_MAX, 1 - TRAIN_VIS_MIN, fill=False, ec=INK, lw=1.6, ls="--"))
        ax.axvline(LOAD_CAP, color=MAGENTA, lw=1.6)
        ax.set_title(f"Attack intensity {inten:.1f}", loc="left", fontsize=12, color=INK)
        ax.set_xlabel("context (never part of an attack signature)")
        ax.grid(False)
    axes[0].set_ylabel("attack signature visibility")
    axes[1].text(0.985, 0.04, "dashed box: the only conditions attacks appeared in during training\nmagenta line: edge of what the agent may explore\norange: detector's 50% decision boundary",
                 ha="right", va="bottom", fontsize=8.2, color=INK2,
                 bbox=dict(facecolor=SURF, edgecolor=GRID, boxstyle="round,pad=0.4"))
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("learned detector: P(attack)")
    fig.suptitle("A learned detector's blind spot follows its training-data gap, not the attack", x=0.01, ha="left", fontsize=14, color=INK)
    fig.savefig(os.path.join(OUT, "ml_detector_map.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 2. blind-spot hunt GIF
def hunt_gif(seed=0, load_slice=0.5):
    env = GridDetectorEnv(noise_seed=seed)
    a = BlindAgent(env, "boundary", step_radius=0.2, seed=seed, label_mode="delta")
    a.bootstrap(n=25, hi=0.6)
    a.calibrate_null()
    n_boot = len(a.X)
    a.explore(60)
    X, y = np.array(a.X), np.array(a.y)

    gx = np.linspace(0, 0.3, 90)
    gy = np.linspace(0.05, 1, 90)
    GX, GY = np.meshgrid(gx, gy)
    G = np.c_[np.full(GX.size, load_slice), GY.ravel(), GX.ravel()]
    truth_blind = ((env.truth(G) - env.truth(np.c_[G[:, 0], np.zeros(len(G)), G[:, 2]])) < DELTA).reshape(GX.shape)

    frames = []
    for k in list(range(n_boot, len(X) + 1, 2)) + [len(X)] * 6:
        sub = BlindAgent(env, "boundary", seed=seed, label_mode="delta")
        sub.null = a.null
        sub.X, sub.y = list(X[:k]), list(y[:k])
        sub.blind_rule()
        pred = sub.predict_blind(G).reshape(GX.shape) if sub.tree is not None else np.zeros(GX.shape, bool)
        impact, blind = sub.labels(X[:k], y[:k])
        fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
        ax.contourf(GX, GY, pred.astype(float), levels=[0.5, 1.5], colors=[ORANGE], alpha=0.18)
        ax.contour(GX, GY, truth_blind.astype(float), levels=[0.5], colors=[INK2], linewidths=1.4, linestyles="--")
        ax.axhline(IMPACT, color=GRID, lw=1)
        old = np.arange(k) < n_boot
        ax.scatter(X[:k][old, 2], X[:k][old, 1], s=22, color=MUTED, alpha=0.5, zorder=3, label="historical data")
        det = ~old & ~blind
        ax.scatter(X[:k][det, 2], X[:k][det, 1], s=40, color=BLUE, edgecolor=SURF, linewidth=1.2, zorder=4,
                   label="experiment: detector responded")
        bl = ~old & blind
        ax.scatter(X[:k][bl, 2], X[:k][bl, 1], s=56, color=ORANGE, edgecolor=SURF, linewidth=1.2, zorder=5,
                   label="experiment: detector blind")
        ax.set_xlim(0, 0.3); ax.set_ylim(0.05, 1)
        ax.set_xlabel("non-stealth fraction (zoom 0–0.3)"); ax.set_ylabel("perturbation magnitude")
        ax.set_title(f"Experiment {max(0, k - n_boot):d} / 60", loc="left", color=INK, fontsize=12)
        ax.text(0.298, 0.08, "dashed = true blind region\nshaded = agent's learned rule", ha="right",
                fontsize=8.5, color=INK2, bbox=dict(facecolor=SURF, edgecolor=GRID, boxstyle="round,pad=0.4"))
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor=SURF, edgecolor=GRID)
        frames.append(frame_to_array(fig))
        plt.close(fig)
    imageio.mimsave(os.path.join(OUT, "blind_spot_hunt.gif"), frames, duration=0.22, loop=0)


# ---------------------------------------------------------------- 3. missing-variable GIF
def missing_gif(seed=3, total=150):
    from scipy import stats
    from run_missing import BATCH, PILOT, interaction_model, px, test_channels  # noqa: E402

    hist = {}
    env = GridContextEnv(noise_seed=seed)
    for strat in ["random", "targeted"]:
        env.rng = np.random.default_rng(seed)
        rng = np.random.default_rng(1000 + seed)
        X, C, y, P, N = [], [], [], [], []

        def go(x):
            y.append(env.run(x)); X.append(x); C.append(env.last_ctx)

        for x in rng.uniform(0, 1, (PILOT, 3)):
            go(x)
        while len(y) <= total:
            P.append(-np.log10(test_channels(np.array(X), np.array(C), np.array(y)))); N.append(len(y))
            if strat == "random":
                nxt = rng.uniform(0, 1, (BATCH, 3))
            else:
                cand = rng.uniform(0, 1, (800, 3))
                m, _ = interaction_model(np.array(X), np.array(C), np.array(y))
                w = m.coef_[px(cand[:1]).shape[1]:].reshape(len(CHANNELS), 4)
                eff = np.abs(np.c_[np.ones(len(cand)), cand] @ w.T).sum(1)
                nxt = cand[np.argsort(-eff)[:BATCH]]
            for x in nxt:
                go(x)
        hist[strat] = (np.array(N), np.array(P))
    thr = -np.log10(0.01 / 3)
    cols = [BLUE, ORANGE, AQUA]
    labels = ["ambient_sensor (the real driver)", "grid_noise (distractor)", "clock (distractor)"]
    ymax = max(10, min(40, max(h[1].max() for h in hist.values()) * 1.1))
    frames = []
    nb = len(hist["random"][0])
    for i in list(range(1, nb + 1)) + [nb] * 6:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, constrained_layout=True)
        for ax, strat in zip(axes, ["random", "targeted"]):
            N, P = hist[strat]
            for j in range(3):
                ax.plot(N[:i], P[:i, j], color=cols[j], lw=2.2, label=labels[j])
                ax.scatter(N[i - 1], P[i - 1, j], s=36, color=cols[j], edgecolor=SURF, zorder=5)
            ax.axhline(thr, color=INK2, lw=1, ls="--")
            ax.text(N[-1], thr + 0.3, "significance threshold", ha="right", fontsize=8.5, color=INK2)
            ax.set_xlim(N[0], N[-1]); ax.set_ylim(0, ymax)
            ax.set_title(f"{strat} experiments", loc="left", fontsize=12, color=INK)
            ax.set_xlabel("experiments run")
        axes[0].set_ylabel("evidence the channel explains\nthe model's error  (−log10 p)")
        axes[1].legend(loc="upper left", fontsize=8.5, facecolor=SURF, edgecolor=GRID)
        fig.suptitle("Finding the variable the model is missing", x=0.01, ha="left", fontsize=14, color=INK)
        frames.append(frame_to_array(fig))
        plt.close(fig)
    imageio.mimsave(os.path.join(OUT, "missing_variable_discovery.gif"), frames, duration=0.35, loop=0)


if __name__ == "__main__":
    which = sys.argv[1:] or ["map", "mlmap", "hunt", "missing"]
    if "map" in which: response_map(); print("map done", flush=True)
    if "mlmap" in which: ml_map(); print("mlmap done", flush=True)
    if "hunt" in which: hunt_gif(); print("hunt done", flush=True)
    if "missing" in which: missing_gif(); print("missing done", flush=True)

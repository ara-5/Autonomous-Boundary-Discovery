"""Turn the saved result files into README charts and a terminal-style image.

    python scripts/make_results.py
Reads results/*.txt (verbatim run output) and writes assets/*.png.
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "assets")
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": ["Segoe UI", "DejaVu Sans"], "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.edgecolor": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def read(name):
    with open(os.path.join(RES, name), encoding="utf-8", errors="replace") as f:
        return f.read().replace("Â±", "±")


# ------------------------------------------------------------- blind-spot chart
def parse_blind(txt):
    blocks, cur = {}, None
    for line in txt.splitlines():
        m = re.match(r"#+ blind-spot type: (\w+)", line)
        if m:
            cur = m.group(1); blocks[cur] = {}; continue
        m = re.match(r"(\w+)\s+TREE seen ([\d.]+) ± ([\d.]+) withheld ([\d.]+) ± ([\d.]+) \| "
                     r"LOGIT seen ([\d.]+) ± ([\d.]+) withheld ([\d.]+) ± ([\d.]+)", line)
        if m and cur:
            v = list(map(float, m.groups()[1:]))
            blocks[cur][m.group(1)] = dict(tree_seen=v[0:2], tree_wh=v[2:4], logit_seen=v[4:6], logit_wh=v[6:8])
    return blocks


def blind_chart():
    blocks = parse_blind(read("blind_spot_results.txt"))
    names = {"uniform": "uniform fuzzing", "random": "random (safe radius)", "uncertainty": "uncertainty sampling",
             "discriminating": "model disagreement", "boundary": "boundary-targeting (ours)"}
    titles = {"delta": "Blind spot: indistinguishable from no perturbation",
              "adaptive": "Blind spot: below load-adaptive alarm threshold"}
    fig, axes = plt.subplots(1, len(blocks), figsize=(11.5, 4.3), sharey=True, constrained_layout=True)
    axes = [axes] if len(blocks) == 1 else axes
    for ax, (kind, rows) in zip(axes, blocks.items()):
        order = [k for k in names if k in rows][::-1]
        for i, k in enumerate(order):
            for key, color, mk, dy in [("logit_seen", BLUE, "o", 0.14), ("logit_wh", ORANGE, "D", -0.14)]:
                m, h = rows[k][key]
                ax.errorbar(m, i + dy, xerr=h, fmt=mk, color=color, ecolor=color, elinewidth=2, capsize=3, ms=7,
                            mec=SURF, mew=1.2)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([names[k] for k in order])
        for t, k in zip(ax.get_yticklabels(), order):
            if k == "boundary":
                t.set_fontweight("bold"); t.set_color(INK)
        ax.set_xlim(0.6, 1.02); ax.xaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
        ax.set_xlabel("balanced accuracy of the learned blind-spot rule (mean ± 95% CI, 20 seeds)")
        ax.set_title(titles[kind], loc="left", fontsize=11.5, color=INK)
    h1 = plt.Line2D([], [], color=BLUE, marker="o", ls="", label="loads the agent explored")
    h2 = plt.Line2D([], [], color=ORANGE, marker="D", ls="", label="withheld high loads (transfer test)")
    axes[0].legend(handles=[h1, h2], loc="lower left", fontsize=8.5, facecolor=SURF, edgecolor=GRID)
    fig.savefig(os.path.join(OUT, "results_blind_spot.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------- missing-variable chart
def missing_chart():
    rows = []
    for line in read("missing_variable_results.txt").splitlines():
        m = re.match(r"(\S+)\s+correct (\d+)/(\d+), wrong channel (\d+); experiments: median (\d+)", line)
        if m:
            rows.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    labels = [r[0] for r in rows][::-1]
    ys = range(len(rows))
    cols = [ORANGE if "targeted" in r[0] else BLUE for r in rows][::-1]
    axes[0].barh(ys, [r[1] for r in rows][::-1], color=cols, height=0.55)
    for y, r in zip(ys, rows[::-1]):
        axes[0].text(r[1] + 0.3, y, f"{r[1]}/{r[2]}   ({r[3]} wrong)", va="center", fontsize=9, color=INK2)
    axes[0].set_xlim(0, 26); axes[0].set_title("Real driver identified (of 20 runs)", loc="left", fontsize=11.5)
    axes[1].barh(ys, [r[4] for r in rows][::-1], color=cols, height=0.55)
    for y, r in zip(ys, rows[::-1]):
        axes[1].text(r[4] + 3, y, str(r[4]), va="center", fontsize=9, color=INK2)
    axes[1].set_xlim(0, 235); axes[1].set_title("Experiments needed (median, cap 200)", loc="left", fontsize=11.5)
    for ax in axes:
        ax.set_yticks(list(ys)); ax.set_yticklabels(labels); ax.xaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    axes[0].text(0, -1.05, "orange = targeted experiment design, blue = random", fontsize=8.5, color=INK2,
                 transform=axes[0].get_yaxis_transform())
    fig.savefig(os.path.join(OUT, "results_missing_variable.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------- terminal-style screenshot
def terminal(name, src, title, max_lines=22):
    lines = [l for l in read(src).splitlines()][:max_lines]
    fig_h = 0.27 * len(lines) + 0.9
    fig = plt.figure(figsize=(12, fig_h))
    fig.patch.set_facecolor("#1a1a19")
    fig.text(0.015, 1 - 0.45 / fig_h, f"$ {title}", color="#0ca30c", family="monospace", fontsize=10, va="center")
    for i, l in enumerate(lines):
        fig.text(0.015, 1 - (0.95 + 0.27 * i) / fig_h, l, color="#f0efec", family="monospace", fontsize=9, va="center")
    fig.savefig(os.path.join(OUT, name), dpi=140, facecolor="#1a1a19")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(RES, "blind_spot_results.txt")):
        blind_chart(); terminal("terminal_blind_spot.png", "blind_spot_results.txt", "python run_blind.py 20", 46)
    missing_chart()
    terminal("terminal_missing_variable.png", "missing_variable_results.txt", "python run_missing.py 20", 10)
    print("results assets done")

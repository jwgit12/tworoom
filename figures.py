"""Thesis figures for the Two-Room LeWM study: environment, data, training, visual probe, rollouts, planning.

Every number in the figures is computed here from the saved dataset, checkpoints, logs and planning results,
and the headline numbers are also written to summary.json.

    python figures.py                       # -> experiments/figures/*.pdf, *.png, summary.json
"""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib import patches  # noqa: E402

import tworoom  # noqa: E402
from lewm import LeWM, LeWMConfig, VisualProbe  # noqa: E402
from probe import door_position, dot_position  # noqa: E402
from train import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

# validated categorical slots 1-3 (all pairs pass the CVD and normal-vision floors), plus a neutral reference
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
PLANNERS = [("cem", "CEM (vanilla LeWM)", BLUE, "o", "-"), ("mcts", "MCTS (predefined actions)", ORANGE, "s", "--"),
            ("oracle", "Oracle (CEM on true game)", GRAY, "^", ":")]
TEXT_WIDTH = 6.3  # inches, an A4 thesis text width

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 1.5,
    "lines.markersize": 4, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved {out / name}.pdf/.png")


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar_exact(a, b):
    """Two-sided exact McNemar test on paired booleans: discordant counts and p-value."""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n if n else 1.0
    return only_a, only_b, min(1.0, 2 * tail)


def parse_train_log(path):
    pat = re.compile(r"epoch\s+(\d+).*train loss ([\d.]+) \(pred ([\d.]+), sigreg ([\d.]+)\).*val pred ([\d.]+) vs copy "
                     r"([\d.]+).*probe R2 emb ([\d.-]+), pred ([\d.-]+) \(x/y/door ([\d.-]+)/([\d.-]+)/([\d.-]+)\)")
    rows = [list(map(float, m.groups())) for m in map(pat.search, Path(path).read_text().splitlines()) if m]
    keys = ["epoch", "loss", "pred", "sigreg", "val_pred", "copy", "r2_emb", "r2_pred", "r2_x", "r2_y", "r2_door"]
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


def parse_probe_log(path):
    pat = re.compile(r"step\s+(\d+).*train mse ([\d.]+) \| val mse ([\d.]+) vs mean frame ([\d.]+)")
    rows = [list(map(float, m.groups())) for m in map(pat.search, Path(path).read_text().splitlines()) if m]
    return {k: [r[i] for r in rows] for i, k in enumerate(["step", "train", "val", "mean_frame"])}


def draw_arena(ax, door=None, door_alpha=1.0):
    """The arena in game coordinates (y down), with the wall and a door gap centred at `door`."""
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor="#ebebe1", edgecolor=MUTED, linewidth=0.8, zorder=0))
    if door is not None:
        lo, hi = door - tworoom.DOOR_HALF, door + tworoom.DOOR_HALF
        for y0, y1 in ((0, lo), (hi, 1)):
            ax.add_patch(patches.Rectangle((tworoom.WALL_X[0], y0), tworoom.WALL_X[1] - tworoom.WALL_X[0], y1 - y0,
                                           facecolor="#28283c", alpha=door_alpha, linewidth=0, zorder=1))
    ax.set(xlim=(-0.02, 1.02), ylim=(1.02, -0.02), aspect="equal")
    ax.grid(False)


def show_frame(ax, frame):
    ax.imshow(frame.permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True), s.set_color("#c3c2b7"), s.set_linewidth(0.5)


def fig_environment(data, val, out):
    """(a) geometry; (b) frames of one held-out episode showing the moving door."""
    fig = plt.figure(figsize=(TEXT_WIDTH, 2.35))
    gs = fig.add_gridspec(2, 6, width_ratios=[2.2, 1, 1, 1, 1, 1], wspace=0.12, hspace=0.25)
    ax = fig.add_subplot(gs[:, 0])
    draw_arena(ax, door=0.5)
    lo, hi = tworoom.DOOR_HALF + tworoom.DOOR_MARGIN, 1 - tworoom.DOOR_HALF - tworoom.DOOR_MARGIN
    ax.add_patch(patches.Rectangle((tworoom.WALL_X[1] + 0.02, lo), 0.015, hi - lo, color=ORANGE, linewidth=0))
    ax.annotate("door centre range\n[%.2f, %.2f]" % (lo, hi), (tworoom.WALL_X[1] + 0.035, 0.5), xytext=(0.62, 0.5),
                va="center", fontsize=6.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.annotate("", (tworoom.WALL_X[0] - 0.02, 0.5 - tworoom.DOOR_HALF), (tworoom.WALL_X[0] - 0.02, 0.5 + tworoom.DOOR_HALF),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.6))
    ax.text(tworoom.WALL_X[0] - 0.035, 0.5, f"door\n{2 * tworoom.DOOR_HALF:.2f}", ha="right", va="center", fontsize=6.5)
    ax.add_patch(patches.Circle((0.2, 0.8), tworoom.RADIUS, color="#dc2828", zorder=3))
    ax.text(0.2, 0.8 - tworoom.RADIUS - 0.03, f"agent\nr = {tworoom.RADIUS}", ha="center", va="bottom", fontsize=6.5)
    ax.add_patch(patches.Circle((0.82, 0.22), tworoom.RADIUS, fill=False, edgecolor=AQUA, linewidth=1.2, zorder=3))
    ax.text(0.82, 0.22 + 0.09, "goal", ha="center", fontsize=6.5)
    ax.text(0.25, 0.05, "left room", ha="center", fontsize=6.5, color=MUTED)
    ax.text(0.75, 0.05, "right room", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xlabel("x"), ax.set_ylabel("y (down)")
    ax.set_xticks([0, 0.5, 1]), ax.set_yticks([0, 0.5, 1])
    ax.set_title("(a) Arena geometry", loc="left")

    # a held-out door-seeking episode that crosses the wall
    seek_val = val[val >= len(data["frames"]) // 2]
    x = data["pos"][seek_val, :, 0] < 0.5
    ep = int(seek_val[(x[:, 1:] != x[:, :-1]).any(1)][0])
    steps = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63]
    for i, t in enumerate(steps):
        a = fig.add_subplot(gs[i // 5, 1 + i % 5])
        show_frame(a, data["frames"][ep, t])
        a.set_title(f"t = {t}", fontsize=6.5, pad=2)
    fig.text(0.355, 0.97, f"(b) Held-out episode {ep}, every 7th raw step t (door speed 0.005 per step)", fontsize=8)
    save(fig, out, "fig1_environment")
    return ep


def fig_data(data, out):
    """(a) one random-walk and one door-seeking path; (b) crossings per episode; (c) door centre over time."""
    E = len(data["frames"])
    seek = torch.arange(E) >= E // 2
    x = data["pos"][..., 0] < 0.5
    crossings = (x[:, 1:] != x[:, :-1]).sum(1)
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.1), gridspec_kw=dict(width_ratios=[1, 1.25, 1.25], wspace=0.45))
    ax = axes[0]
    draw_arena(ax, door=None)
    ax.axvspan(*tworoom.WALL_X, color="#28283c", alpha=0.18, linewidth=0)
    for ep, color, label, ls in ((int((~seek).nonzero()[10]), BLUE, "random walk", "-"),
                                 (int(seek.nonzero()[1]), ORANGE, "door-seeking", "--")):
        path = data["pos"][ep]
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.2, ls=ls, label=f"{label} (ep {ep})")
        ax.plot(*path[0], marker="o", color=color, ms=4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=1, fontsize=6.5)
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("(a) Example agent paths", loc="left")
    ax.text(0.5, 0.03, "wall column", ha="center", va="top", fontsize=6, color=MUTED)
    ax.text(0.03, 0.96, "dot = start", va="bottom", fontsize=6, color=MUTED)

    ax = axes[1]
    values = torch.arange(int(crossings.max()) + 1)
    for i, (mask, color, label, hatch) in enumerate(((~seek, BLUE, "random walk", None), (seek, ORANGE, "door-seeking", "///"))):
        counts = torch.stack([(crossings[mask] == v).sum() for v in values])
        ax.bar(values + (i - 0.5) * 0.4, counts, width=0.38, color=color, hatch=hatch, edgecolor="white", linewidth=0.5,
               label=f"{label} (n = {int(mask.sum())})")
    ax.set(xlabel="wall crossings per episode", ylabel="episodes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    ax.set_title("(b) Crossings per episode", loc="left")

    ax = axes[2]
    t = torch.arange(data["door"].size(1))
    for i, (ep, color, ls) in enumerate(((0, BLUE, "-"), (1, ORANGE, "--"), (2, AQUA, ":"))):
        ax.plot(t, data["door"][ep], color=color, ls=ls, label=f"episode {ep}")
    lo, hi = tworoom.DOOR_HALF + tworoom.DOOR_MARGIN, 1 - tworoom.DOOR_HALF - tworoom.DOOR_MARGIN
    for v in (lo, hi):
        ax.axhline(v, color=GRAY, lw=0.8, ls="-.")
    ax.set(xlabel="raw step t", ylabel="door centre (y, down)", ylim=(0.95, 0.05))
    ax.text(64, lo - 0.02, "range limits", ha="right", fontsize=6, color=MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=6, handlelength=1.8, columnspacing=0.8)
    ax.set_title("(c) Door centre over time", loc="left")
    save(fig, out, "fig2_data")
    return {"episodes": E, "crossing_fraction_all": float((crossings > 0).float().mean()),
            "crossing_fraction_random_walk": float((crossings[~seek] > 0).float().mean()),
            "crossing_fraction_door_seeking": float((crossings[seek] > 0).float().mean()),
            "mean_crossings_random_walk": float(crossings[~seek].float().mean()),
            "mean_crossings_door_seeking": float(crossings[seek].float().mean())}


def fig_training(log, out):
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.0), gridspec_kw=dict(wspace=0.45))
    e = log["epoch"]
    ax = axes[0]
    ax.plot(e, log["pred"], color=BLUE, marker="o", label="train prediction loss")
    ax.plot(e, log["val_pred"], color=ORANGE, marker="s", ls="--", label="val prediction loss")
    ax.plot(e, log["copy"], color=GRAY, marker="^", ls=":", label='val "nothing moves" baseline')
    ax.set(yscale="log", xlabel="epoch", ylabel="MSE in latent space")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
    ax.set_title("(a) Next-embedding prediction", loc="left")
    ax = axes[1]
    ax.plot(e, log["sigreg"], color=BLUE, marker="o")
    ax.set(xlabel="epoch", ylabel="SIGReg statistic (train)")
    ax.set_title("(b) SIGReg regularizer", loc="left")
    ax = axes[2]
    for key, color, mk, ls, label in (("r2_x", BLUE, "o", "-", "agent x"), ("r2_y", ORANGE, "s", "--", "agent y"),
                                      ("r2_door", AQUA, "^", ":", "door centre")):
        ax.plot(e, log[key], color=color, marker=mk, ls=ls, label=label)
    ax.set(xlabel="epoch", ylabel="linear probe $R^2$ (val)", ylim=(0.95, 1.003))
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
    ax.set_title("(c) State in the embedding", loc="left")
    for ax in axes:
        ax.set_xticks(e[::2] if len(e) > 6 else e)
    save(fig, out, "fig3_training")


def fig_probe(frames, recon, probe_log, out):
    """(a) held-out frames and reconstructions; (b) probe training curve."""
    n = len(frames)
    fig = plt.figure(figsize=(TEXT_WIDTH, 2.35))
    gs = fig.add_gridspec(2, n + 2, width_ratios=[1] * n + [1.6, 3.2], wspace=0.08, hspace=0.12)
    for i in range(n):
        for r, img in enumerate((frames[i], recon[i])):
            ax = fig.add_subplot(gs[r, i])
            show_frame(ax, img)
            if i == 0:
                ax.set_ylabel(["real", "decoded"][r], fontsize=7)
            if r == 1:
                mse = (frames[i].float() / 255 - recon[i].float() / 255).square().mean().item()
                ax.set_xlabel(f"{mse * 1e3:.2f}", fontsize=6)
    fig.text(0.13, 0.0, "pixel MSE ×10$^{-3}$ per decoded frame ([0, 1] RGB)", fontsize=6.5, color=MUTED)
    fig.text(0.125, 0.97, "(a) Held-out frames and their reconstructions from the latent", fontsize=8)
    ax = fig.add_subplot(gs[:, n + 1])
    ax.plot(probe_log["step"], probe_log["train"], color=BLUE, label="train")
    ax.plot(probe_log["step"], probe_log["val"], color=ORANGE, ls="--", label="validation")
    ax.axhline(probe_log["mean_frame"][-1], color=GRAY, ls=":", label="mean-frame baseline")
    ax.set(yscale="log", xlabel="probe training step", ylabel="MSE (normalized pixels)")
    ax.legend(fontsize=6, loc="center right")
    ax.set_title("(b) Probe training", loc="left")
    save(fig, out, "fig4_probe")


def fig_rollout_examples(real, imagined, eps, context, fs, out):
    """Real and decoded imagined frames, with the absolute difference, for a few held-out episodes."""
    K = real.size(1)
    rows = 3 * len(eps)
    fig, axes = plt.subplots(rows, K, figsize=(TEXT_WIDTH, 0.52 * rows + 0.35), gridspec_kw=dict(wspace=0.05, hspace=0.08))
    for j, e in enumerate(eps):
        diff = (real[e].float() - imagined[e].float()).abs().mean(1)  # (K, H, W)
        for k in range(K):
            show_frame(axes[3 * j, k], real[e, k])
            show_frame(axes[3 * j + 1, k], imagined[e, k])
            ax = axes[3 * j + 2, k]
            ax.imshow(diff[k].cpu().numpy(), cmap="Greys", vmin=0, vmax=128, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True), sp.set_color("#c3c2b7"), sp.set_linewidth(0.5)
            if k < context:
                for r in range(3):
                    for s in axes[3 * j + r, k].spines.values():
                        s.set_color(AQUA), s.set_linewidth(1.2)
            if j == 0:
                axes[0, k].set_title("obs %d" % k if k < context else f"+{k - context + 1}", fontsize=6.5, pad=2)
        axes[3 * j, 0].set_ylabel(f"ep {e}\nreal", fontsize=6.5)
        axes[3 * j + 1, 0].set_ylabel("imagined", fontsize=6.5)
        axes[3 * j + 2, 0].set_ylabel("|diff|", fontsize=6.5)
    fig.suptitle(f"Latent rollouts decoded by the probe: {context} observed frames (green border), then +k imagined "
                 f"model steps of {fs} raw actions.\n|diff|: mean absolute RGB difference, black = 128 of 255 or more",
                 fontsize=7.5, y=0.955)
    save(fig, out, "fig5_rollout_examples")


def band(ax, x, values, color, ls, mk, label):
    """Median line with an interquartile band; values: (episodes, steps), NaNs ignored."""
    q = torch.nanquantile(values, torch.tensor([0.25, 0.5, 0.75]), dim=0)
    ax.fill_between(x, q[0], q[2], color=color, alpha=0.18, linewidth=0)
    ax.plot(x, q[1], color=color, ls=ls, marker=mk, label=label)
    return q[1]


def fig_rollout_error(metrics, context, out):
    K = metrics["pixel_imagined"].size(1)
    x = torch.arange(K)
    metrics = dict(metrics)
    for key in ("latent_copy", "dot_stay", "door_stay"):  # baselines start at the last observed frame
        m = metrics[key].clone()
        m[:, : context - 1] = float("nan")
        if key == "latent_copy":
            m[:, context - 1] = float("nan")  # exactly 0 there, which a log axis cannot show
        metrics[key] = m
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 4.6), gridspec_kw=dict(wspace=0.32, hspace=0.95))
    specs = [
        (axes[0, 0], "(a) Pixel error of decoded frames", "pixel MSE ([0, 1] RGB)",
         [("pixel_imagined", BLUE, "-", "o", "imagined (rollout → probe)"),
          ("pixel_encoded", AQUA, "--", "s", "encoded real frame → probe")], True),
        (axes[0, 1], "(b) Latent prediction error", "squared distance in latent",
         [("latent_imagined", BLUE, "-", "o", "imagined vs. encoded real"),
          ("latent_copy", GRAY, ":", "^", "last observed embedding")], True),
        (axes[1, 0], "(c) Agent position from decoded frames", "position error",
         [("dot_imagined", BLUE, "-", "o", "imagined"), ("dot_stay", GRAY, ":", "^", "agent stops")], False),
        (axes[1, 1], "(d) Door centre from decoded frames", "door centre error",
         [("door_imagined", BLUE, "-", "o", "imagined"), ("door_stay", GRAY, ":", "^", "door stops")], False),
    ]
    for ax, title, ylabel, series, log in specs:
        for key, color, ls, mk, label in series:
            band(ax, x, metrics[key], color, ls, mk, label)
        ax.axvspan(-0.4, context - 0.6, color=AQUA, alpha=0.08, linewidth=0)
        ax.text((context - 1) / 2, 0.03, "observed", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6, color=MUTED)
        ax.set(xlabel="model step k (5 raw actions each)", ylabel=ylabel, title=title, xlim=(-0.4, K - 0.6))
        ax.title.set_ha("left"), ax.title.set_position((0, 1))
        if log:
            ax.set_yscale("log")
        ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
    for ax in (axes[1, 0], axes[1, 1]):
        ax.axhline(tworoom.RADIUS, color=MUTED, lw=0.6, ls="-.")
        ax.text(K - 0.7, tworoom.RADIUS, "agent radius", ha="right", va="bottom", fontsize=6, color=MUTED)
    fig.text(0.5, -0.03, "Lines: median over held-out episodes. Bands: interquartile range.", ha="center", fontsize=6.5,
             color=MUTED)
    save(fig, out, "fig6_rollout_error")


def fig_planning(results, out, max_steps):
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.25), gridspec_kw=dict(width_ratios=[1.35, 1, 1], wspace=0.4))
    ax = axes[0]
    groups = [("all", None), ("same room", False), ("door", True)]
    width = 0.26
    for i, (key, label, color, mk, ls) in enumerate(PLANNERS):
        r = results[key]
        for g, (_, cross) in enumerate(groups):
            mask = torch.ones_like(r["cross"]) if cross is None else r["cross"] == cross
            k, n = int(r["reached"][mask].sum()), int(mask.sum())
            lo, hi = wilson(k, n)
            xg = g + (i - 1) * width
            ax.bar(xg, k / n, width=width * 0.9, color=color, label=label if g == 0 else None,
                   hatch=[None, "///", ".."][i], edgecolor="white", linewidth=0.5)
            ax.errorbar(xg, k / n, yerr=[[k / n - lo], [hi - k / n]], color=INK, capsize=2, lw=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n(n = {int((torch.ones_like(results['cem']['cross']) if c is None else results['cem']['cross'] == c).sum())})"
                        for g, c in groups])
    ax.set(ylabel="success rate", ylim=(0, 1.05))
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    ax.set_title("(a) Success rate, 95% Wilson CI", loc="left")

    ax = axes[1]
    t = torch.arange(max_steps + 1)
    for key, label, color, mk, ls in PLANNERS:
        r = results[key]
        cum = [(r["steps_to_goal"] <= s).float().mean().item() for s in t]
        ax.step(t, cum, where="post", color=color, ls=ls, label=label)
    ax.set(xlabel="model steps used", ylabel="fraction of tasks solved", ylim=(0, 1.02))
    ax.set_title("(b) Cumulative success", loc="left")

    ax = axes[2]
    for key, label, color, mk, ls in PLANNERS:
        d, _ = results[key]["final_distance"].sort()
        ax.step(d, torch.arange(1, len(d) + 1) / len(d), where="post", color=color, ls=ls, label=label)
    ax.axvline(0.05, color=MUTED, lw=0.6, ls="-.")
    ax.text(0.052, 0.04, "success\nthreshold", fontsize=6, color=MUTED)
    ax.set(xscale="log", xlabel="final distance to goal", ylabel="fraction of tasks", ylim=(0, 1.02))
    ax.set_title("(c) Final distance (ECDF)", loc="left")
    save(fig, out, "fig7_planning")


def fig_planning_examples(results, out, n=4):
    """The same tasks planned by CEM and MCTS: door tasks first, preferring ones where the two planners differ."""
    cem, mcts = results["cem"], results["mcts"]
    differ = (cem["reached"] != mcts["reached"]) & cem["cross"]
    order = torch.cat([differ.nonzero().squeeze(1), (cem["cross"] & ~differ).nonzero().squeeze(1)])[:n]
    fig, axes = plt.subplots(2, n, figsize=(TEXT_WIDTH, 3.4), gridspec_kw=dict(wspace=0.08, hspace=0.25))
    for j, task in enumerate(order.tolist()):
        for i, (key, label, color, mk, ls) in enumerate(PLANNERS[:2]):
            r = results[key]
            ax = axes[i, j]
            steps = r["steps_to_goal"][task]
            T = int(steps) if not torch.isnan(steps) else r["path"].size(1) - 1
            door = r["door"][task]
            draw_arena(ax, door=float(door[T]))
            ax.axvspan(*tworoom.WALL_X, ymin=0, ymax=1, color="#28283c", alpha=0.08, linewidth=0)
            path = r["path"][task, : T + 1]
            ax.plot(path[:, 0], path[:, 1], color=color, lw=1.2, marker=mk, ms=2.5)
            ax.add_patch(patches.Circle(tuple(r["start"][task].tolist()), tworoom.RADIUS, fill=False, edgecolor=INK, lw=0.8))
            ax.add_patch(patches.Circle(tuple(r["goal"][task].tolist()), tworoom.RADIUS, fill=False, edgecolor=AQUA, lw=1.2))
            ok = bool(r["reached"][task])
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"task {task}: {'reached, %d steps' % T if ok else 'missed, %d steps' % T}", fontsize=6.5)
            if j == 0:
                ax.set_ylabel(label, fontsize=7)
    fig.text(0.5, 0.02, "Black ring: start. Green ring: goal. Wall drawn at the last shown step; shaded column: "
             "door sweep.", ha="center", fontsize=6.5, color=MUTED)
    save(fig, out, "fig8_planning_examples")


@torch.no_grad()
def rollout_metrics(model, probe, data, val, ckpt, context, steps, device, batch=200):
    """Decode rollouts for all held-out episodes and measure them against the real frames."""
    fs = ckpt["frameskip"]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    to_pixels = lambda f: (f.float() / 255 - mean) / std
    to_uint8 = lambda x: ((x.float() * std + mean).clamp(0, 1) * 255).round().to(torch.uint8)
    idx = torch.arange(steps + 1) * fs
    out = {k: [] for k in ("pixel_imagined", "pixel_encoded", "latent_imagined", "latent_copy", "dot_imagined",
                           "dot_stay", "door_imagined", "door_stay")}
    reals, imags = [], []
    for i in range(0, len(val), batch):
        eps = val[i : i + batch]
        real = data["frames"][eps][:, idx].to(device)  # (B, K, 3, H, W)
        acts = data["actions"][eps][:, : steps * fs].to(device)
        acts = ((acts - ckpt["action_mean"].to(device)) / ckpt["action_std"].to(device)).view(len(eps), steps, fs * 2)
        emb = model.rollout(to_pixels(real[:, :context]), acts)  # (B, K, D): context encoded, the rest imagined
        enc = model.encode(to_pixels(real))  # every real frame encoded
        imagined = to_uint8(probe(emb))
        encoded = to_uint8(probe(enc))
        f = lambda u: u.float() / 255
        out["pixel_imagined"].append((f(imagined) - f(real)).square().mean((2, 3, 4)))
        out["pixel_encoded"].append((f(encoded) - f(real)).square().mean((2, 3, 4)))
        out["latent_imagined"].append((emb - enc).square().sum(-1))
        out["latent_copy"].append((enc[:, context - 1 : context] - enc).square().sum(-1))
        B, K = real.shape[:2]
        pos = data["pos"][eps][:, idx].to(device)
        door = data["door"][eps][:, idx].to(device)
        out["dot_imagined"].append((dot_position(imagined.flatten(0, 1)).view(B, K, 2) - pos).norm(dim=-1))
        out["dot_stay"].append((pos[:, context - 1 : context] - pos).norm(dim=-1))
        out["door_imagined"].append((door_position(imagined.flatten(0, 1)).view(B, K) - door).abs())
        out["door_stay"].append((door[:, context - 1 : context] - door).abs())
        reals.append(real.cpu()), imags.append(imagined.cpu())
    out = {k: torch.cat(v).float().cpu() for k, v in out.items()}
    return out, torch.cat(reals), torch.cat(imags)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tworoom_6000_moving_slow_seek.pt")
    p.add_argument("--ckpt", default="checkpoints/lewm_tworoom_6000_moving_slow_seek.pt")
    p.add_argument("--probe", default="checkpoints/probe_tworoom_6000_moving_slow_seek_emb.pt")
    p.add_argument("--train-log", default="experiments/train_6000_moving_slow_seek/train.log")
    p.add_argument("--probe-log", default="experiments/probe_6000_moving_slow_seek_emb/probe.log")
    p.add_argument("--plans", default="experiments", help="folder with plan_cem/, plan_mcts/, plan_oracle/")
    p.add_argument("--out", default="experiments/figures")
    p.add_argument("--context", type=int, default=3)
    p.add_argument("--rollout-steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0, help="train.py's seed, to recover its validation split")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load(args.data)
    torch.manual_seed(args.seed)  # the same episode split as train.py and probe.py
    perm = torch.randperm(len(data["frames"]))
    val = perm[: max(1, int(len(perm) * 0.1))]
    summary = {"validation_episodes": len(val)}

    summary["environment_example_episode"] = fig_environment(data, val, out)
    summary["data"] = fig_data(data, out)
    log = parse_train_log(args.train_log)
    fig_training(log, out)
    summary["training"] = {k: v[-1] for k, v in log.items()}

    ckpt = torch.load(args.ckpt)
    model = LeWM(LeWMConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    state = torch.load(args.probe)
    probe = VisualProbe(in_dim=state["in_dim"], img_size=state["img_size"], patch_size=state["patch_size"])
    probe.load_state_dict(state["probe"])
    probe = probe.to(device).eval()
    summary["model"] = {"config": ckpt["config"], "frameskip": ckpt["frameskip"],
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "probe_parameters": sum(p.numel() for p in probe.parameters())}

    # probe reconstructions of single held-out frames
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    frames = data["frames"][val[:: len(val) // 6][:6], 32].to(device)
    with torch.no_grad():
        z = model.encode(((frames.float() / 255 - mean) / std)[:, None])[:, 0]
        recon = ((probe(z) * std + mean).clamp(0, 1) * 255).round().to(torch.uint8)
    probe_log = parse_probe_log(args.probe_log)
    fig_probe(frames.cpu(), recon.cpu(), probe_log, out)
    summary["probe"] = {"val_mse_normalized": probe_log["val"][-1], "mean_frame_mse_normalized": probe_log["mean_frame"][-1]}

    metrics, real, imagined = rollout_metrics(model, probe, data, val, ckpt, args.context, args.rollout_steps, device)
    median = lambda v: torch.nanquantile(v, 0.5, dim=0).tolist()
    summary["rollout"] = {k: median(v) for k, v in metrics.items()}
    summary["rollout"]["dot_missing_fraction"] = metrics["dot_imagined"].isnan().float().mean(0).tolist()
    summary["rollout"]["door_missing_fraction"] = metrics["door_imagined"].isnan().float().mean(0).tolist()
    pos = data["pos"][val][:, :: ckpt["frameskip"]][:, : args.rollout_steps + 1, 0] < 0.5
    crosses = (pos[:, 1:] != pos[:, :-1]).any(1)
    fig_rollout_examples(real, imagined, crosses.nonzero().squeeze(1)[:3].tolist(), args.context, ckpt["frameskip"], out)
    fig_rollout_error(metrics, args.context, out)

    results = {}
    for key, *_ in PLANNERS:
        path = Path(args.plans) / f"plan_{key}" / "results.pt"
        if path.exists():
            results[key] = torch.load(path)
    if all(k in results for k, *_ in PLANNERS):
        fig_planning(results, out, results["cem"]["args"]["max_steps"])
        fig_planning_examples(results, out)
        plan_summary = {}
        for key, *_ in PLANNERS:
            r = results[key]
            row = {"seconds_per_episode": r["seconds_per_episode"], "args": r["args"],
                   "median_final_distance": float(r["final_distance"].median()),
                   "mean_steps_to_goal_successes": float(r["steps_to_goal"].nanmean())}
            fail = ~r["reached"]
            closest = (r["path"][fail] - r["goal"][fail][:, None]).norm(dim=-1).min(1).values
            row["failures"] = {"count": int(fail.sum()), "same_room": int((fail & ~r["cross"]).sum()),
                               "door": int((fail & r["cross"]).sum()),
                               "closest_approach_median": float(closest.median()) if len(closest) else None,
                               "closest_approach_below_0.10": int((closest < 0.10).sum()),
                               "final_distance_median": float(r["final_distance"][fail].median()) if len(closest) else None}
            for name, mask in (("all", torch.ones_like(r["cross"])), ("same_room", ~r["cross"]), ("door", r["cross"])):
                k, n = int(r["reached"][mask].sum()), int(mask.sum())
                row[name] = {"success": k, "tasks": n, "rate": k / n, "wilson95": wilson(k, n)}
            plan_summary[key] = row
        for name, mask in (("all", None), ("same_room", False), ("door", True)):
            m = torch.ones_like(results["cem"]["cross"]) if mask is None else results["cem"]["cross"] == mask
            only_cem, only_mcts, pval = mcnemar_exact(results["cem"]["reached"][m], results["mcts"]["reached"][m])
            plan_summary.setdefault("mcnemar_cem_vs_mcts", {})[name] = {
                "only_cem_solved": only_cem, "only_mcts_solved": only_mcts, "p_value": pval}
        summary["planning"] = plan_summary
    else:
        print("planning results missing; skipped figures 7 and 8")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out / 'summary.json'}")


if __name__ == "__main__":
    main()

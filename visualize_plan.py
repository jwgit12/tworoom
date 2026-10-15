"""Follow the LeWM planner through one Two-Room task and decode what it imagined at every replanning step.

At each step the planner picks its best action plan. Two versions of that plan are drawn side by side:
- imagined: LeWM's predicted embeddings for the plan (`model.rollout`), decoded to frames by an emb VisualProbe
- real: the same planned actions executed in the true game from the same position
Arrows show the planned action of each model step. Only the first step is executed, then the planner replans.

    python visualize_plan.py                 # the first door task of plan.py's 50 tasks
    python visualize_plan.py --task 7        # any task index
"""

import argparse
from pathlib import Path

import torch

import tworoom
from lewm import LeWM, LeWMConfig, VisualProbe
from plan import cem, lewm_cost, to_pixels
from probe import dot_position, labeled_grid
from train import IMAGENET_MEAN, IMAGENET_STD

TILE = 128  # pixels per frame in the timeline
PANEL = 256  # pixels per panel in the overview
GOAL = (40, 170, 70)
EXECUTED = (240, 140, 140)
IMAGINED = (40, 90, 220)
REAL = (240, 150, 0)
ARROW = (0, 0, 0)


def disc(img, center, r, color):
    """Fill a disc of radius r (pixels) at center (x, y) in pixels into a uint8 (H, W, 3) image."""
    H, W, _ = img.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    img[(xx - center[0]).square() + (yy - center[1]).square() <= r * r] = torch.tensor(color, dtype=torch.uint8)


def ring(img, center, r, color, width=2):
    H, W, _ = img.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    d = ((xx - center[0]).square() + (yy - center[1]).square()).sqrt()
    img[(d <= r) & (d > r - width)] = torch.tensor(color, dtype=torch.uint8)


def line(img, a, b, color, width=1):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    for t in torch.linspace(0, 1, n + 1):
        x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        x0, y0 = int(x) - width // 2, int(y) - width // 2
        img[max(0, y0) : max(0, y0 + width), max(0, x0) : max(0, x0 + width)] = torch.tensor(color, dtype=torch.uint8)


def arrow(img, start, end, color=ARROW, width=2):
    """An arrow from start to end, both (x, y) in pixels."""
    d = torch.tensor([end[0] - start[0], end[1] - start[1]], dtype=torch.float)
    length = d.norm()
    if length < 2:  # no visible motion: mark the spot instead
        disc(img, start, 2, color)
        return
    line(img, start, end, color, width)
    head = min(6.0, 0.5 * length.item())
    for angle in (2.6, -2.6):  # two barbs, about 150 degrees back from the direction
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        tip = (d[0] * c - d[1] * s, d[0] * s + d[1] * c)
        tip = (end[0] + head * tip[0] / length, end[1] + head * tip[1] / length)
        line(img, end, tip, color, width)


def px(pos, size):
    return (pos[0].item() * size, pos[1].item() * size)


def raw_actions(plan, fs):
    """plan: (H, K) with K = 2 (held) or fs * 2 (varied) -> (H, fs, 2) raw actions per model step."""
    return plan.reshape(plan.size(0), -1, 2).expand(plan.size(0), fs, 2)


def simulate(pos, plan, fs):
    """Run a plan in the true game from pos (2,) -> positions after each model step, (H + 1, 2)."""
    out, p = [pos], pos[None]
    for step_actions in raw_actions(plan, fs):
        for a in step_actions:
            p = tworoom.step(p, a[None])
        out.append(p[0])
    return torch.stack(out)


def frame_with_marks(frame, pos, goal, action, fs):
    """uint8 (3, S, S) frame -> (TILE, TILE, 3) with the goal ring and the planned move as an arrow."""
    img = frame.repeat_interleave(TILE // frame.size(-1), -1).repeat_interleave(TILE // frame.size(-2), -2)
    img = img.permute(1, 2, 0).contiguous()
    ring(img, px(goal, TILE), tworoom.RADIUS * TILE + 1, GOAL)
    if pos is not None and action is not None:
        # where the planned action would move the dot in free space during this model step
        move = action.mean(0) * tworoom.SPEED * fs
        start = px(pos, TILE)
        arrow(img, start, (start[0] + move[0].item() * TILE, start[1] + move[1].item() * TILE))
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/lewm_tworoom_3000.pt")
    p.add_argument("--probe", default="checkpoints/probe_tworoom_3000_emb.pt")
    p.add_argument("--task", type=int, default=None, help="task index (default: the first door task)")
    p.add_argument("--tasks", type=int, default=50, help="as plan.py --episodes, so task indices match")
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--elites", type=int, default=30)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--vary-actions", action="store_true")
    p.add_argument("--stride", type=int, default=2, help="timeline: draw every stride-th planned step")
    p.add_argument("--show-steps", type=int, default=6, help="timeline: replanning steps drawn, evenly spaced")
    p.add_argument("--outdir", default=None, help="default experiments/plan_scenario_task<k>")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt)
    model = LeWM(LeWMConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    fs = ckpt["frameskip"]
    action_mean, action_std = ckpt["action_mean"].to(device), ckpt["action_std"].to(device)

    state = torch.load(args.probe)
    assert state["space"] == "emb", "rollouts live in the emb space: pass a probe trained with --space emb"
    probe = VisualProbe(in_dim=state["in_dim"], img_size=state["img_size"], patch_size=state["patch_size"])
    probe.load_state_dict(state["probe"])
    probe = probe.to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    to_uint8 = lambda pixels: ((pixels.float() * std + mean).clamp(0, 1) * 255).round().to(torch.uint8)

    # the same tasks as plan.py
    g = torch.Generator().manual_seed(args.seed)
    starts, goals = tworoom.random_start(args.tasks, g), tworoom.random_start(args.tasks, g)
    cross = (starts[:, 0] < 0.5) != (goals[:, 0] < 0.5)
    task = args.task if args.task is not None else int(cross.nonzero()[0])
    pos, goal = starts[task].clone(), goals[task]
    print(f"task {task}: start ({pos[0]:.2f}, {pos[1]:.2f}) -> goal ({goal[0]:.2f}, {goal[1]:.2f}), "
          f"{'through the door' if cross[task] else 'same room'}")

    with torch.no_grad():
        goal_emb = model.encode(to_pixels(tworoom.render(goal[None]).to(device)))[:, 0]

    plan = torch.zeros(1, args.horizon, 2 * fs if args.vary_actions else 2, device=device)
    steps = []  # one record per replanning step
    executed = [pos.clone()]
    for t in range(args.max_steps):
        obs = to_pixels(tworoom.render(pos[None]).to(device))
        cost = lewm_cost(model, obs, goal_emb, action_mean, action_std, fs)
        best = cem(cost, plan, args.samples, args.elites, args.iters)  # (1, H, K)

        with torch.no_grad():
            actions = (raw_actions(best[0], fs) - action_mean) / action_std  # (H, fs, 2)
            emb = model.rollout(obs, actions.reshape(1, args.horizon, fs * 2))  # (1, H + 1, D)
            imagined = to_uint8(probe(emb[0])).cpu()  # (H + 1, 3, S, S): now, then each planned step
            plan_cost = cost(best[:, None])[0, 0].item()
        real_pos = simulate(pos, best[0].cpu(), fs)
        steps.append({
            "t": t, "pos": pos.clone(), "plan": best[0].cpu(), "cost": plan_cost,
            "imagined": imagined, "imagined_pos": dot_position(imagined), "real_pos": real_pos,
        })

        pos = real_pos[1]  # execute only the first planned step
        executed.append(pos.clone())
        plan = torch.cat([best[:, 1:], torch.zeros_like(best[:, :1])], 1)
        dist = (pos - goal).norm().item()
        drift = (steps[-1]["imagined_pos"][1:] - real_pos[1:]).norm(dim=-1)
        print(f"step {t:2d}: cost {plan_cost:7.2f} | distance to goal {dist:.3f} | "
              f"imagined vs real plan: drift {drift.nanmean():.3f} mean, {drift.nan_to_num(0).max():.3f} max")
        if dist < args.threshold:
            break
    reached = dist < args.threshold
    print(f"{'reached the goal' if reached else 'did not reach the goal'} after {len(steps)} steps")

    outdir = args.outdir or f"experiments/plan_scenario_task{task}"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    executed = torch.stack(executed)
    save_overview(f"{outdir}/overview.png", steps, executed, goal, task, reached, fs)
    save_timeline(f"{outdir}/timeline.png", steps, goal, fs, args)
    print(f"saved {outdir}/overview.png and {outdir}/timeline.png")


def save_overview(path, steps, executed, goal, task, reached, fs, cols=5, gap=6):
    """One panel per replanning step: the plan as imagined (blue) and as it really plays out (orange)."""
    panels = []
    for i, s in enumerate(steps):
        img = tworoom.render(s["pos"][None], PANEL)[0].permute(1, 2, 0).contiguous()
        for a, b in zip(executed[: i], executed[1 : i + 1]):  # path executed so far
            line(img, px(a, PANEL), px(b, PANEL), EXECUTED, 3)
        ring(img, px(goal, PANEL), tworoom.RADIUS * PANEL, GOAL, 3)
        for track, color in ((s["real_pos"], REAL), (s["imagined_pos"], IMAGINED)):
            pts = [px(q, PANEL) for q in track if not q.isnan().any()]
            for a, b in zip(pts, pts[1:]):
                line(img, a, b, color, 2)
            for q in pts[1:]:
                disc(img, q, 3, color)
        start = px(s["pos"], PANEL)
        move = raw_actions(s["plan"], fs)[0].mean(0) * tworoom.SPEED * fs
        arrow(img, start, (start[0] + move[0].item() * PANEL, start[1] + move[1].item() * PANEL), ARROW, 3)
        label = f"step {s['t']}  cost {s['cost']:.1f}"
        img[2:20, 2 : 10 + tworoom.text_width(label, 2)] = 255  # keep the label readable over the wall
        tworoom.draw_text(img, label, 6, 6, (0, 0, 0), 2)
        panels.append(img)

    rows = (len(panels) + cols - 1) // cols
    header = 64
    out = torch.full((header + rows * (PANEL + gap), cols * (PANEL + gap) + gap, 3), 255, dtype=torch.uint8)
    for i, img in enumerate(panels):
        r, c = divmod(i, cols)
        y, x = header + r * (PANEL + gap), gap + c * (PANEL + gap)
        out[y : y + PANEL, x : x + PANEL] = img
    title = f"task {task}: {'reached the goal' if reached else 'missed the goal'} in {len(steps)} steps"
    tworoom.draw_text(out, title, gap, 8, (0, 0, 0), 3)
    legend = [("imagined plan", IMAGINED), ("same actions in the real game", REAL),
              ("executed path", EXECUTED), ("goal", GOAL), ("first action", ARROW)]
    x = gap
    for text, color in legend:
        out[36 : 36 + 14, x : x + 14] = torch.tensor(color, dtype=torch.uint8)
        tworoom.draw_text(out, text, x + 20, 38, (0, 0, 0), 2)
        x += 20 + tworoom.text_width(text, 2) + 24
    tworoom.save_png(path, out)


def save_timeline(path, steps, goal, fs, args):
    """Per shown replanning step: the decoded imagined plan above the real outcome of the same actions."""
    shown = torch.linspace(0, len(steps) - 1, min(args.show_steps, len(steps))).round().long().unique().tolist()
    cols = list(range(0, args.horizon + 1, args.stride))
    rows, labels = [], []
    for i in shown:
        s = steps[i]
        acts = raw_actions(s["plan"], fs)
        imag, real = [], []
        for k in cols:
            a = acts[k] if k < args.horizon else None
            imag.append(frame_with_marks(s["imagined"][k], s["imagined_pos"][k], goal, a, fs))
            real_frame = tworoom.render(s["real_pos"][k][None])[0]
            real.append(frame_with_marks(real_frame, s["real_pos"][k], goal, a, fs))
        rows += [torch.stack(imag).permute(0, 3, 1, 2), torch.stack(real).permute(0, 3, 1, 2)]
        labels += [f"step {s['t']} imagined", f"step {s['t']} real"]
    col_labels = ["now"] + [f"+{k}" for k in cols[1:]]
    title = (f"planned {args.horizon} model steps ahead, one column per {args.stride}. "
             f"arrow: planned action, green ring: goal")
    tworoom.save_png(path, labeled_grid(rows, labels, col_labels, title, zoom=1))


if __name__ == "__main__":
    main()

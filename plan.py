"""Reach a goal frame in Two-Room by planning with a trained LeWM.

Model-predictive control with the cross-entropy method (CEM): sample action plans, imagine where they
lead with `model.rollout`, and refit the sampling distribution to the plans whose imagined embeddings stay
closest to the goal frame's embedding. Execute the first step of the best plan in the game, observe, replan.

The cost sums the distance to the goal over every planned step, not just the last one: with a final-step
cost, any path that ends at the goal is equally good, so the executed first action is arbitrary and the
agent wanders.

    python plan.py                      # LeWM planner on 50 random start/goal pairs
    python plan.py --planner oracle     # same CEM on the true game: separates model errors from planner limits
"""

import argparse
import time

import torch

import tworoom
from lewm import LeWM, LeWMConfig
from train import IMAGENET_MEAN, IMAGENET_STD

GOAL_COLOR = (40, 170, 70)
TRAIL_COLOR = (240, 140, 140)


def to_pixels(frames):
    """uint8 frames (N, 3, H, W) -> normalized (N, 1, 3, H, W), as in training."""
    mean = torch.tensor(IMAGENET_MEAN, device=frames.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=frames.device).view(3, 1, 1)
    return ((frames.float() / 255 - mean) / std).unsqueeze(1)


@torch.no_grad()
def cem(cost_fn, mean, samples, elites, iters):
    """Minimize cost_fn over plans. mean: (B, H, 2) warm start; cost_fn: (B, N, H, 2) -> (B, N)."""
    std = torch.ones_like(mean)
    for _ in range(iters):
        noise = torch.randn(mean.size(0), samples, *mean.shape[1:], device=mean.device)
        plans = (mean[:, None] + std[:, None] * noise).clamp(-1, 1)
        plans[:, 0] = mean.clamp(-1, 1)  # keep the current best guess in the population
        best = cost_fn(plans).topk(elites, dim=1, largest=False).indices
        elite = plans.gather(1, best[..., None, None].expand(-1, -1, *plans.shape[2:]))
        mean, std = elite.mean(1), elite.std(1)
    return mean


def lewm_cost(model, obs, goal_emb, action_mean, action_std, frameskip, max_batch=600):
    """Squared distance between the imagined embeddings and the goal embedding, averaged over the plan.

    obs: (B, 1, C, H, W) current frames, goal_emb: (B, D). Each planned step holds 2 dims (one action
    for all `frameskip` raw steps) or frameskip * 2 (one per raw step); either way the model gets its
    action_dim = frameskip * 2 after normalization.
    """

    def cost(plans):
        B, N, H, _ = plans.shape
        actions = plans.reshape(B, N, H, -1, 2).expand(B, N, H, frameskip, 2)
        actions = ((actions - action_mean) / action_std).reshape(B, N, H, frameskip * 2)
        per_call = max(1, max_batch // N)  # rollout gets slow on MPS beyond ~600 sequences per call
        imagined = []
        for i in range(0, B, per_call):
            pixels = obs[i : i + per_call, None].expand(-1, N, *obs.shape[1:]).flatten(0, 1)
            imagined.append(model.rollout(pixels, actions[i : i + per_call].flatten(0, 1))[:, 1:])
        imagined = torch.cat(imagined).view(B, N, H, -1)  # predicted embedding after each planned step
        return (imagined - goal_emb[:, None, None]).square().sum(-1).mean(-1)

    return cost


def oracle_cost(pos, goal, frameskip):
    """Euclidean distance to the goal, averaged over the plan, simulating each plan in the real game."""

    def cost(plans):
        B, N, H, _ = plans.shape
        p = pos[:, None].expand(B, N, 2).reshape(B * N, 2)
        actions = plans.cpu().reshape(B * N, H, -1, 2).expand(B * N, H, frameskip, 2)
        total = 0
        for h in range(H):
            for k in range(frameskip):
                p = tworoom.step(p, actions[:, h, k])
            total = total + (p.view(B, N, 2) - goal[:, None]).norm(dim=-1)
        return (total / H).to(plans.device)

    return cost


def draw_episode(path, goal, size=160):
    """One frame: where the dot ended up, the trail it took (pink), and the goal (green)."""
    frame = tworoom.render(path[-1:], size)[0]
    c = (torch.arange(size) + 0.5) / size
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    near = lambda p, r: (xx - p[0]).square() + (yy - p[1]).square() < r**2
    agent = near(path[-1], tworoom.RADIUS)
    trail = torch.stack([near(p, 0.012) for p in path]).any(0) & ~agent
    img = frame.permute(1, 2, 0)  # a view: writing into it draws on frame
    img[near(goal, tworoom.RADIUS) & ~agent] = torch.tensor(GOAL_COLOR, dtype=torch.uint8)
    img[trail] = torch.tensor(TRAIL_COLOR, dtype=torch.uint8)
    return frame


def save_trajectories(path, history, goal, rows=2, cols=5, gap=4):
    """A grid with one tile per episode."""
    history = torch.stack(history, 1)  # (E, T, 2)
    n = min(rows * cols, len(goal))
    tiles = torch.stack([draw_episode(history[i], goal[i]) for i in range(n)])
    tiles = torch.cat([tiles, torch.full((rows * cols - n, *tiles.shape[1:]), 255, dtype=torch.uint8)])
    tiles = torch.nn.functional.pad(tiles, (gap, gap, gap, gap), value=255)
    _, C, S, _ = tiles.shape
    tworoom.save_png(path, tiles.view(rows, cols, C, S, S).permute(0, 3, 1, 4, 2).reshape(rows * S, cols * S, C))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/lewm_tworoom.pt")
    p.add_argument("--planner", choices=["lewm", "oracle"], default="lewm")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=25, help="budget in model steps (frameskip raw actions each)")
    p.add_argument("--horizon", type=int, default=10, help="planning horizon in model steps")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--elites", type=int, default=30)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.05, help="success distance (one dot radius)")
    p.add_argument("--save-png", default="plans.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vary-actions", action="store_true",
                   help="plan one action per raw step instead of holding one for all frameskip steps")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt)
    model = LeWM(LeWMConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    fs = ckpt["frameskip"]
    action_mean, action_std = ckpt["action_mean"].to(device), ckpt["action_std"].to(device)

    g = torch.Generator().manual_seed(args.seed)
    pos, goal = tworoom.random_start(args.episodes, g), tworoom.random_start(args.episodes, g)
    cross = (pos[:, 0] < 0.5) != (goal[:, 0] < 0.5)
    with torch.no_grad():
        goal_emb = model.encode(to_pixels(tworoom.render(goal).to(device)))[:, 0]

    plan = torch.zeros(args.episodes, args.horizon, 2 * fs if args.vary_actions else 2, device=device)
    reached = torch.zeros(args.episodes, dtype=torch.bool)
    steps_to_goal = torch.full((args.episodes,), float("nan"))
    history = [pos.clone()]
    t0 = time.time()
    for t in range(1, args.max_steps + 1):
        idx = (~reached).nonzero().squeeze(1)
        if len(idx) == 0:
            break
        if args.planner == "lewm":
            obs = to_pixels(tworoom.render(pos[idx]).to(device))
            cost = lewm_cost(model, obs, goal_emb[idx], action_mean, action_std, fs)
        else:
            cost = oracle_cost(pos[idx], goal[idx], fs)
        best = cem(cost, plan[idx], args.samples, args.elites, args.iters)

        action = best[:, 0].cpu().reshape(len(idx), -1, 2).expand(len(idx), fs, 2)
        for k in range(fs):
            pos[idx] = tworoom.step(pos[idx], action[:, k])
        plan[idx] = torch.cat([best[:, 1:], torch.zeros_like(best[:, :1])], 1)  # warm start: shift by one step

        done = ~reached & ((pos - goal).norm(dim=-1) < args.threshold)
        steps_to_goal[done] = t
        reached |= done
        history.append(pos.clone())
        print(f"step {t:2d}: {int(reached.sum())}/{args.episodes} at goal", end="\r", flush=True)

    print()
    dist = (pos - goal).norm(dim=-1)
    rate = lambda mask: f"{reached[mask].float().mean():.0%} (n={int(mask.sum())})"
    print(f"planner={args.planner} horizon={args.horizon} vary_actions={args.vary_actions} budget={args.max_steps} steps "
          f"({args.max_steps * fs} raw actions), CEM {args.samples} samples x {args.iters} iters")
    print(f"success {rate(torch.ones_like(cross))} | same room {rate(~cross)} | through the door {rate(cross)}")
    print(f"steps to goal (successes) {steps_to_goal.nanmean():.1f} | final distance median {dist.median():.3f} | "
          f"{(time.time() - t0) / args.episodes:.1f}s per episode")
    if args.save_png:
        save_trajectories(args.save_png, history, goal)
        print(f"saved the first episodes to {args.save_png} (green = goal, pink = path taken)")


if __name__ == "__main__":
    main()

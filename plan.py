"""Reach a goal frame in Two-Room by planning with a trained LeWM.

Model-predictive control with the cross-entropy method (CEM): sample action plans, imagine where they
lead with `model.rollout`, and refit the sampling distribution to the plans whose imagined embeddings stay
closest to the goal frame's embedding. Execute the first step of the best plan in the game, observe, replan.

The cost sums the distance to the goal over every planned step, not just the last one: with a final-step
cost, any path that ends at the goal is equally good, so the executed first action is arbitrary and the
agent wanders.

    python plan.py                      # LeWM planner on 50 random start/goal pairs
    python plan.py --planner oracle     # same CEM on the true game: separates model errors from planner limits
    python plan.py --door moving --context 3    # a door that moves; 3 frames of context show its direction
"""

import argparse
import math
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


def lewm_cost(model, obs, goal_emb, action_mean, action_std, frameskip, past_actions=None, max_batch=600):
    """Squared distance between the imagined embeddings and the goal embedding, averaged over the plan.

    obs: (B, t0, C, H, W) the last t0 observed frames, the current one last; goal_emb: (B, D).
    past_actions: (B, t0 - 1, frameskip * 2) normalized actions taken between those frames, or None for t0 = 1.
    Each planned 2D action is held for `frameskip` raw steps, which gives the model's action_dim = frameskip * 2.
    """
    t0 = obs.size(1)

    def cost(plans):
        B, N, H, _ = plans.shape
        actions = plans[..., None, :].expand(B, N, H, frameskip, 2)
        actions = ((actions - action_mean) / action_std).reshape(B, N, H, frameskip * 2)
        if t0 > 1:  # the context frames come with the actions taken between them
            actions = torch.cat([past_actions[:, None].expand(B, N, -1, -1), actions], 2)
        per_call = max(1, max_batch // N)  # rollout gets slow on MPS beyond ~600 sequences per call
        imagined = []
        for i in range(0, B, per_call):
            pixels = obs[i : i + per_call, None].expand(-1, N, *obs.shape[1:]).flatten(0, 1)
            imagined.append(model.rollout(pixels, actions[i : i + per_call].flatten(0, 1))[:, t0:])
        imagined = torch.cat(imagined).view(B, N, H, -1)  # predicted embedding after each planned step
        return (imagined - goal_emb[:, None, None]).square().sum(-1).mean(-1)

    return cost


class Node:
    """A state in LeWM's imagined future: the last embeddings and the actions between them, as the predictor sees them."""

    __slots__ = ("embs", "acts", "depth", "ret", "children", "N", "W")

    def __init__(self, embs, acts, depth, ret):
        self.embs, self.acts, self.depth, self.ret = embs, acts, depth, ret  # ret: summed reward from the root
        self.children, self.N, self.W = None, 0, 0.0


@torch.no_grad()
def mcts(model, obs, past_actions, goal_emb, actions_raw, action_mean, action_std, frameskip, horizon, sims, c_uct):
    """Monte Carlo tree search over predefined actions in LeWM's latent space, one tree per task.

    Each edge holds one of `actions_raw` for a whole model step (frameskip raw steps). The reward of a state is
    minus its squared distance to the goal embedding. A path is valued like a CEM plan: the mean reward over
    the horizon, assuming the dot stays at the leaf for the remaining steps. Expanding a leaf predicts all its
    children in one batched predictor call, and each child starts with one visit at its own value; the best
    child's value is backed up. Selection uses UCT on values normalized per tree. Returns the most visited
    first action per task, (B, 2).
    """
    B, H = obs.size(0), model.config.history_size
    K = len(actions_raw)
    actions_norm = ((actions_raw[:, None].expand(K, frameskip, 2) - action_mean) / action_std).reshape(K, -1)
    root_embs = model.encode(obs)  # (B, t0, D)
    no_acts = root_embs.new_zeros(0, actions_norm.size(1))
    roots = [Node(root_embs[b], past_actions[b] if past_actions is not None else no_acts, 0, 0.0) for b in range(B)]
    lo, hi = [math.inf] * B, [-math.inf] * B  # value range seen per tree, to normalize Q for UCT

    def select(node, b):
        log_n = math.log(node.N)
        span = max(hi[b] - lo[b], 1e-9)
        return max(node.children, key=lambda ch: (ch.W / ch.N - lo[b]) / span + c_uct * math.sqrt(log_n / ch.N))

    for _ in range(sims):
        paths = []
        for b in range(B):
            node, path = roots[b], [roots[b]]
            while node.children is not None:
                node = select(node, b)
                path.append(node)
            paths.append(path)

        # expand every leaf that is not at the horizon, grouped by history length so they batch together
        values = [None] * B
        groups = {}
        for b, path in enumerate(paths):
            leaf = path[-1]
            if leaf.depth < horizon:
                groups.setdefault(len(leaf.embs), []).append(b)
            else:
                values[b] = leaf.ret / horizon
        for tasks in groups.values():
            leaves = [paths[b][-1] for b in tasks]
            n = len(tasks)
            embs = torch.stack([leaf.embs for leaf in leaves])  # (n, L, D)
            acts = torch.stack([leaf.acts for leaf in leaves])  # (n, L - 1, A)
            emb_in = embs[:, None].expand(n, K, *embs.shape[1:]).flatten(0, 1)
            act_in = torch.cat([acts[:, None].expand(n, K, *acts.shape[1:]),
                                actions_norm[None, :, None].expand(n, K, 1, -1)], 2)
            nxt = model.predict(emb_in, act_in.flatten(0, 1))[:, -1].view(n, K, -1)  # (n, K, D)
            reward = -(nxt - goal_emb[tasks][:, None]).square().sum(-1)  # (n, K)
            for i, b in enumerate(tasks):
                leaf = leaves[i]
                depth = leaf.depth + 1
                children = []
                for k in range(K):
                    r = reward[i, k].item()
                    child = Node(torch.cat([leaf.embs, nxt[i, k : k + 1]])[-H:],
                                 torch.cat([leaf.acts, actions_norm[k : k + 1]])[-(H - 1):], depth, leaf.ret + r)
                    child.N, child.W = 1, (child.ret + (horizon - depth) * r) / horizon
                    lo[b], hi[b] = min(lo[b], child.W), max(hi[b], child.W)
                    children.append(child)
                leaf.children = children
                values[b] = max(ch.W for ch in children)

        for b, path in enumerate(paths):
            for node in path:
                node.N += 1
                node.W += values[b]

    best = [max(range(K), key=lambda k: (root.children[k].N, root.children[k].W / root.children[k].N)) for root in roots]
    return actions_raw[torch.tensor(best)]


def mcts_actions(directions):
    """Standing still plus `directions` moves, evenly spaced in angle -> (K, 2).

    Each move reaches the edge of the action square [-1, 1]^2, so diagonals are as fast as CEM's corners.
    """
    angle = torch.arange(directions) * 2 * math.pi / directions
    moves = torch.stack([angle.cos(), angle.sin()], 1)
    moves = (moves / moves.abs().amax(1, keepdim=True)).round(decimals=6)
    return torch.cat([torch.zeros(1, 2), moves])


def oracle_cost(pos, goal, frameskip, door=None, direction=None, door_speed=0.0):
    """Euclidean distance to the goal, averaged over the plan, simulating each plan in the real game."""

    def cost(plans):
        B, N, H, _ = plans.shape
        p = pos[:, None].expand(B, N, 2).reshape(B * N, 2)
        repeat = lambda v: None if v is None else v[:, None].expand(B, N).reshape(B * N)
        d, dirn = repeat(door), repeat(direction)
        actions = plans.cpu().reshape(B * N, H, 1, 2).expand(B * N, H, frameskip, 2)
        total = 0
        for h in range(H):
            for k in range(frameskip):
                p, d, dirn = tworoom.world_step(p, actions[:, h, k], d, dirn, door_speed)
            total = total + (p.view(B, N, 2) - goal[:, None]).norm(dim=-1)
        return (total / H).to(plans.device)

    return cost


def draw_episode(path, goal, size=160, door=None):
    """One frame: where the dot ended up, the trail it took (pink), and the goal (green)."""
    frame = tworoom.render(path[-1:], size, door)[0]
    c = (torch.arange(size) + 0.5) / size
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    near = lambda p, r: (xx - p[0]).square() + (yy - p[1]).square() < r**2
    agent = near(path[-1], tworoom.RADIUS)
    trail = torch.stack([near(p, 0.012) for p in path]).any(0) & ~agent
    img = frame.permute(1, 2, 0)  # a view: writing into it draws on frame
    img[near(goal, tworoom.RADIUS) & ~agent] = torch.tensor(GOAL_COLOR, dtype=torch.uint8)
    img[trail] = torch.tensor(TRAIL_COLOR, dtype=torch.uint8)
    return frame


def save_trajectories(path, history, goal, door=None, rows=2, cols=5, gap=4):
    """A grid with one tile per episode, drawn with each episode's final door."""
    history = torch.stack(history, 1)  # (E, T, 2)
    n = min(rows * cols, len(goal))
    door_of = lambda i: None if door is None else door[i : i + 1]
    tiles = torch.stack([draw_episode(history[i], goal[i], door=door_of(i)) for i in range(n)])
    tiles = torch.cat([tiles, torch.full((rows * cols - n, *tiles.shape[1:]), 255, dtype=torch.uint8)])
    tiles = torch.nn.functional.pad(tiles, (gap, gap, gap, gap), value=255)
    _, C, S, _ = tiles.shape
    tworoom.save_png(path, tiles.view(rows, cols, C, S, S).permute(0, 3, 1, 4, 2).reshape(rows * S, cols * S, C))


def sample_tasks(n, g, door, rooms_only):
    """Start or goal positions; with rooms_only, none inside the wall's column, where a moving door could trap it."""
    pos = tworoom.random_start(n, g, door)
    in_column = lambda q: (q[:, 0] + tworoom.RADIUS > tworoom.WALL_X[0]) & (q[:, 0] - tworoom.RADIUS < tworoom.WALL_X[1])
    bad = in_column(pos) if rooms_only else torch.zeros(n, dtype=torch.bool)
    while bad.any():
        pos[bad] = tworoom.random_start(int(bad.sum()), g, None if door is None else door[bad])
        bad = in_column(pos)
    return pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/lewm_tworoom.pt")
    p.add_argument("--planner", choices=["lewm", "oracle", "mcts"], default="lewm",
                   help="lewm: CEM on the LeWM cost; oracle: CEM on the true game; mcts: tree search on LeWM")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=25, help="budget in model steps (frameskip raw actions each)")
    p.add_argument("--horizon", type=int, default=10, help="planning horizon in model steps")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--elites", type=int, default=30)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.05, help="success distance (one dot radius)")
    p.add_argument("--save-png", default="plans.png")
    p.add_argument("--save-results", default=None, help="save every task's outcome and path to this .pt file")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--door", choices=["fixed", "moving"], default="fixed",
                   help="fixed: the default door; moving: random start height and direction, then it moves")
    p.add_argument("--door-speed", type=float, default=0.01, help="with --door moving: door movement per raw step")
    p.add_argument("--context", type=int, default=1,
                   help="observed frames given to the model (one per model step, up to its history size)")
    p.add_argument("--sims", type=int, default=200, help="mcts: simulations per replanning step")
    p.add_argument("--uct-c", type=float, default=1.0, help="mcts: exploration constant on normalized values")
    p.add_argument("--directions", type=int, default=8, help="mcts: predefined move directions, plus standing still")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt)
    model = LeWM(LeWMConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    fs = ckpt["frameskip"]
    action_mean, action_std = ckpt["action_mean"].to(device), ckpt["action_std"].to(device)

    g = torch.Generator().manual_seed(args.seed)
    door = direction = None
    speed = args.door_speed if args.door == "moving" else 0.0
    if args.door == "moving":  # fixed draws nothing extra, so its tasks match those from before the option existed
        door = tworoom.random_door(args.episodes, g)
        direction = torch.where(torch.rand(args.episodes, generator=g) < 0.5, -1.0, 1.0)
    pos = sample_tasks(args.episodes, g, door, rooms_only=args.door == "moving")
    goal = sample_tasks(args.episodes, g, door, rooms_only=args.door == "moving")
    cross = (pos[:, 0] < 0.5) != (goal[:, 0] < 0.5)
    with torch.no_grad():  # the goal frame shows the door where it starts
        goal_emb = model.encode(to_pixels(tworoom.render(goal, door=door).to(device)))[:, 0]
    context = max(1, min(args.context, model.config.history_size))
    frames = [tworoom.render(pos, door=door).to(device)]  # observed frames, one per model step
    past = []  # normalized actions executed between them, (E, fs * 2) each

    start, start_door, start_direction = pos.clone(), door, direction
    plan = torch.zeros(args.episodes, args.horizon, 2, device=device)
    reached = torch.zeros(args.episodes, dtype=torch.bool)
    steps_to_goal = torch.full((args.episodes,), float("nan"))
    history, door_history = [pos.clone()], [door]
    t0 = time.time()
    for t in range(1, args.max_steps + 1):
        idx = (~reached).nonzero().squeeze(1)
        if len(idx) == 0:
            break
        sub = lambda v: None if v is None else v[idx]
        if args.planner != "oracle":
            c = min(context, len(frames))
            obs = torch.stack([f[idx.to(device)] for f in frames[-c:]], 1)  # (n, c, 3, H, W)
            obs = to_pixels(obs.flatten(0, 1)).view(len(idx), c, *obs.shape[2:])
            past_actions = torch.stack([a[idx.to(device)] for a in past[-(c - 1):]], 1) if c > 1 else None
        if args.planner == "mcts":
            action = mcts(model, obs, past_actions, goal_emb[idx], mcts_actions(args.directions).to(device),
                          action_mean, action_std, fs, args.horizon, args.sims, args.uct_c).cpu()[:, None]
            action = action.expand(len(idx), fs, 2)
        else:
            if args.planner == "lewm":
                cost = lewm_cost(model, obs, goal_emb[idx], action_mean, action_std, fs, past_actions)
            else:
                cost = oracle_cost(pos[idx], goal[idx], fs, sub(door), sub(direction), speed)
            best = cem(cost, plan[idx], args.samples, args.elites, args.iters)
            action = best[:, 0].cpu()[:, None].expand(len(idx), fs, 2)
            plan[idx] = torch.cat([best[:, 1:], torch.zeros_like(best[:, :1])], 1)  # warm start: shift by one step
        p_idx, d_idx, dir_idx = pos[idx], sub(door), sub(direction)
        for k in range(fs):
            p_idx, d_idx, dir_idx = tworoom.world_step(p_idx, action[:, k], d_idx, dir_idx, speed)
        pos[idx] = p_idx
        for _ in range(fs if speed > 0 else 0):  # every door moves on, also for tasks already at the goal
            door, direction = tworoom.move_door(door, direction, speed)
        executed = torch.zeros(args.episodes, fs * 2, device=device)
        executed[idx.to(device)] = ((action.to(device) - action_mean) / action_std).reshape(len(idx), fs * 2)
        past.append(executed)
        frames.append(tworoom.render(pos, door=door).to(device))

        done = ~reached & ((pos - goal).norm(dim=-1) < args.threshold)
        steps_to_goal[done] = t
        reached |= done
        history.append(pos.clone())
        door_history.append(door)
        print(f"step {t:2d}: {int(reached.sum())}/{args.episodes} at goal", end="\r", flush=True)

    print()
    dist = (pos - goal).norm(dim=-1)
    rate = lambda mask: f"{reached[mask].float().mean():.0%} (n={int(mask.sum())})"
    print(f"planner={args.planner} door={args.door} context={context} horizon={args.horizon} "
          f"budget={args.max_steps} steps ({args.max_steps * fs} raw actions), "
          + (f"MCTS {args.sims} sims, {args.directions + 1} predefined actions, c={args.uct_c}"
             if args.planner == "mcts" else f"CEM {args.samples} samples x {args.iters} iters"))
    print(f"success {rate(torch.ones_like(cross))} | same room {rate(~cross)} | through the door {rate(cross)}")
    seconds = (time.time() - t0) / args.episodes
    print(f"steps to goal (successes) {steps_to_goal.nanmean():.1f} | final distance median {dist.median():.3f} | "
          f"{seconds:.1f}s per episode")
    if args.save_results:
        # paths keep their last position once at the goal, so all have max_steps + 1 entries
        while len(history) < args.max_steps + 1:
            history.append(pos.clone())
            if speed > 0:
                for _ in range(fs):
                    door, direction = tworoom.move_door(door, direction, speed)
            door_history.append(door)
        torch.save({
            "args": vars(args), "frameskip": fs, "context": context,
            "start": start, "goal": goal, "cross": cross, "reached": reached, "steps_to_goal": steps_to_goal,
            "final_distance": dist, "path": torch.stack(history, 1),
            "door": torch.stack(door_history, 1) if start_door is not None else None,
            "start_door": start_door, "start_direction": start_direction, "seconds_per_episode": seconds,
        }, args.save_results)
        print(f"saved per-task results to {args.save_results}")
    if args.save_png:
        save_trajectories(args.save_png, history, goal, door)
        print(f"saved the first episodes to {args.save_png} (green = goal, pink = path taken)")


if __name__ == "__main__":
    main()

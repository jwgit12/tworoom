"""Two-Room: a tiny point-navigation game (a toy version of the Two-Room task from the LeWM paper).

A red dot moves in a square arena split by a vertical wall with a single door. The action is a
2D velocity in [-1, 1]^2. Everything is batched torch, so generating a dataset is fast.

    python tworoom.py --episodes 400 --steps 64 --out data/tworoom.pt
"""

import argparse
import struct
import zlib
from pathlib import Path

import torch

# arena geometry, in [0, 1] coordinates
RADIUS = 0.05
SPEED = 0.025  # max displacement per step
WALL_X = (0.47, 0.53)
DOOR_Y = (0.38, 0.62)

BACKGROUND = (235, 235, 225)
WALL_COLOR = (40, 40, 60)
AGENT_COLOR = (220, 40, 40)


def is_free(pos):
    """pos: (N, 2) -> (N,) bool, True where a dot at pos touches neither the border nor the wall."""
    x, y = pos.unbind(-1)
    in_bounds = (x >= RADIUS) & (x <= 1 - RADIUS) & (y >= RADIUS) & (y <= 1 - RADIUS)
    overlaps_wall_x = (x + RADIUS > WALL_X[0]) & (x - RADIUS < WALL_X[1])
    in_door = (y - RADIUS >= DOOR_Y[0]) & (y + RADIUS <= DOOR_Y[1])
    return in_bounds & ~(overlaps_wall_x & ~in_door)


def step(pos, action):
    """Move by action * SPEED. On collision, slide along the free axis, else stay put."""
    new = pos + action.clamp(-1, 1) * SPEED
    only_x = torch.stack([new[:, 0], pos[:, 1]], -1)
    only_y = torch.stack([pos[:, 0], new[:, 1]], -1)
    out = torch.where(is_free(only_y)[:, None], only_y, pos)
    out = torch.where(is_free(only_x)[:, None], only_x, out)
    return torch.where(is_free(new)[:, None], new, out)


def render(pos, size=64):
    """pos: (N, 2) -> uint8 frames (N, 3, size, size)."""
    coords = (torch.arange(size, device=pos.device) + 0.5) / size
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    wall = (xx > WALL_X[0]) & (xx < WALL_X[1]) & ~((yy > DOOR_Y[0]) & (yy < DOOR_Y[1]))
    agent = (xx - pos[:, 0, None, None]).square() + (yy - pos[:, 1, None, None]).square() < RADIUS**2

    img = torch.tensor(BACKGROUND, dtype=torch.uint8, device=pos.device).view(1, 3, 1, 1)
    img = img.expand(pos.size(0), 3, size, size).clone()
    img[:, :, wall] = torch.tensor(WALL_COLOR, dtype=torch.uint8, device=pos.device)[:, None]
    img.permute(0, 2, 3, 1)[agent] = torch.tensor(AGENT_COLOR, dtype=torch.uint8, device=pos.device)
    return img


def save_png(path, img):
    """Write a uint8 (H, W, 3) tensor as a PNG, without extra dependencies."""
    h, w, _ = img.shape
    rows = torch.cat([torch.zeros(h, 1, dtype=torch.uint8), img.reshape(h, w * 3)], 1)  # filter byte per row

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    data = zlib.compress(bytes(rows.flatten().tolist()))
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", data) + chunk(b"IEND", b""))


def random_start(n, generator=None):
    """Uniformly sample n free positions by rejection."""
    pos = torch.empty(n, 2)
    todo = torch.ones(n, dtype=torch.bool)
    while todo.any():
        pos[todo] = torch.rand(int(todo.sum()), 2, generator=generator)
        todo = ~is_free(pos)
    return pos


def generate(episodes, steps, size=64, seed=0):
    """Roll out a smooth random-walk policy. Returns a dict of tensors:

    frames  (E, steps + 1, 3, size, size) uint8
    actions (E, steps, 2) float, actions[:, t] moves frames[:, t] to frames[:, t + 1]
    pos     (E, steps + 1, 2) float, ground-truth dot position (only used for evaluation probes)
    """
    g = torch.Generator().manual_seed(seed)
    pos = random_start(episodes, g)
    action = torch.rand(episodes, 2, generator=g) * 2 - 1
    frames, actions, positions = [render(pos, size)], [], [pos]
    for _ in range(steps):
        # a random walk in action space: smooth enough to travel across rooms and through the door
        action = (action + 0.4 * torch.randn(episodes, 2, generator=g)).clamp(-1, 1)
        pos = step(pos, action)
        frames.append(render(pos, size))
        actions.append(action)
        positions.append(pos)
    return {
        "frames": torch.stack(frames, 1),
        "actions": torch.stack(actions, 1),
        "pos": torch.stack(positions, 1),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/tworoom.pt")
    args = p.parse_args()

    data = generate(args.episodes, args.steps, args.size, args.seed)
    crossed = ((data["pos"][..., 0] < 0.5).any(1) & (data["pos"][..., 0] > 0.5).any(1)).float().mean()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out)
    mb = data["frames"].numel() / 2**20
    print(f"saved {args.episodes} episodes x {args.steps} steps to {args.out} ({mb:.0f} MB of frames)")
    print(f"{crossed:.0%} of episodes pass through the door")

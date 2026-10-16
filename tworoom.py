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
DOOR_Y = (0.38, 0.62)  # the default door; with --moving-door it starts at a random height and moves
DOOR_HALF = (DOOR_Y[1] - DOOR_Y[0]) / 2
DOOR_MARGIN = 0.05  # keep at least this much wall above and below a random door

BACKGROUND = (235, 235, 225)
WALL_COLOR = (40, 40, 60)
AGENT_COLOR = (220, 40, 40)


def door_bounds(pos, door):
    """door: (N,) door centers, or None for the default door -> (lo, hi), each broadcastable to pos[:, 0]."""
    if door is None:
        return DOOR_Y
    return door.to(pos.device) - DOOR_HALF, door.to(pos.device) + DOOR_HALF


def random_door(n, generator=None):
    """Sample n door centers uniformly, keeping DOOR_MARGIN of wall above and below each door."""
    lo = DOOR_HALF + DOOR_MARGIN
    return lo + torch.rand(n, generator=generator) * (1 - 2 * lo)


def move_door(door, direction, speed):
    """Slide door centers by direction * speed, bouncing between the highest and lowest allowed heights."""
    lo, hi = DOOR_HALF + DOOR_MARGIN, 1 - DOOR_HALF - DOOR_MARGIN
    new = door + direction * speed
    direction = torch.where((new < lo) | (new > hi), -direction, direction)
    return new.clamp(lo, hi), direction


def carry(pos, door):
    """Push a dot that the moving wall now overlaps back into the door gap, so the wall carries it along."""
    eps = 1e-6  # keep clear of the gap's edges so is_free holds despite rounding
    y = torch.maximum(torch.minimum(pos[:, 1], door + DOOR_HALF - RADIUS - eps), door - DOOR_HALF + RADIUS + eps)
    return torch.where(is_free(pos, door)[:, None], pos, torch.stack([pos[:, 0], y], -1))


def world_step(pos, action, door=None, direction=None, door_speed=0.0):
    """One raw step of the whole game: move a moving door (carrying the dot), then the dot."""
    if door_speed > 0:
        door, direction = move_door(door, direction, door_speed)
        pos = carry(pos, door)
    return step(pos, action, door), door, direction


def is_free(pos, door=None):
    """pos: (N, 2) -> (N,) bool, True where a dot at pos touches neither the border nor the wall."""
    x, y = pos.unbind(-1)
    door_lo, door_hi = door_bounds(pos, door)
    in_bounds = (x >= RADIUS) & (x <= 1 - RADIUS) & (y >= RADIUS) & (y <= 1 - RADIUS)
    overlaps_wall_x = (x + RADIUS > WALL_X[0]) & (x - RADIUS < WALL_X[1])
    in_door = (y - RADIUS >= door_lo) & (y + RADIUS <= door_hi)
    return in_bounds & ~(overlaps_wall_x & ~in_door)


def step(pos, action, door=None):
    """Move by action * SPEED. On collision, slide along the free axis, else stay put."""
    new = pos + action.clamp(-1, 1) * SPEED
    only_x = torch.stack([new[:, 0], pos[:, 1]], -1)
    only_y = torch.stack([pos[:, 0], new[:, 1]], -1)
    out = torch.where(is_free(only_y, door)[:, None], only_y, pos)
    out = torch.where(is_free(only_x, door)[:, None], only_x, out)
    return torch.where(is_free(new, door)[:, None], new, out)


def render(pos, size=64, door=None):
    """pos: (N, 2), door: (N,) door centers or None -> uint8 frames (N, 3, size, size)."""
    coords = (torch.arange(size, device=pos.device) + 0.5) / size
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    door_lo, door_hi = door_bounds(pos, door)
    if door is not None:
        door_lo, door_hi = door_lo[:, None, None], door_hi[:, None, None]
    wall = (xx > WALL_X[0]) & (xx < WALL_X[1]) & ~((yy > door_lo) & (yy < door_hi))
    wall = wall.expand(pos.size(0), size, size)
    agent = (xx - pos[:, 0, None, None]).square() + (yy - pos[:, 1, None, None]).square() < RADIUS**2

    img = torch.tensor(BACKGROUND, dtype=torch.uint8, device=pos.device).view(1, 3, 1, 1)
    img = img.expand(pos.size(0), 3, size, size).clone()
    img.permute(0, 2, 3, 1)[wall] = torch.tensor(WALL_COLOR, dtype=torch.uint8, device=pos.device)
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


# a 3x5 bitmap font for labeling images without extra dependencies, rows separated by spaces
FONT = {
    "A": ".#. #.# ### #.# #.#", "B": "##. #.# ##. #.# ##.", "C": ".## #.. #.. #.. .##", "D": "##. #.# #.# #.# ##.",
    "E": "### #.. ##. #.. ###", "F": "### #.. ##. #.. #..", "G": ".## #.. #.# #.# .##", "H": "#.# #.# ### #.# #.#",
    "I": "### .#. .#. .#. ###", "J": "..# ..# ..# #.# .#.", "K": "#.# #.# ##. #.# #.#", "L": "#.. #.. #.. #.. ###",
    "M": "#.# ### ### #.# #.#", "N": "##. #.# #.# #.# #.#", "O": ".#. #.# #.# #.# .#.", "P": "##. #.# ##. #.. #..",
    "Q": ".#. #.# #.# ##. .##", "R": "##. #.# ##. #.# #.#", "S": ".## #.. .#. ..# ##.", "T": "### .#. .#. .#. .#.",
    "U": "#.# #.# #.# #.# ###", "V": "#.# #.# #.# #.# .#.", "W": "#.# #.# ### ### #.#", "X": "#.# #.# .#. #.# #.#",
    "Y": "#.# #.# .#. .#. .#.", "Z": "### ..# .#. #.. ###", "0": "### #.# #.# #.# ###", "1": ".#. ##. .#. .#. ###",
    "2": "##. ..# .#. #.. ###", "3": "##. ..# .#. ..# ##.", "4": "#.# #.# ### ..# ..#", "5": "### #.. ##. ..# ##.",
    "6": ".## #.. ### #.# ###", "7": "### ..# .#. .#. .#.", "8": "### #.# ### #.# ###", "9": "### #.# ### ..# ##.",
    "=": "... ### ... ### ...", "+": "... .#. ### .#. ...", "-": "... ... ### ... ...", ".": "... ... ... ... .#.",
    ":": "... .#. ... .#. ...", "/": "..# ..# .#. #.. #..", ",": "... ... ... .#. #..", "(": ".#. #.. #.. #.. .#.",
    ")": ".#. ..# ..# ..# .#.", " ": "... ... ... ... ...",
}


def draw_text(img, text, x, y, color=(0, 0, 0), scale=2):
    """Draw text into a uint8 (H, W, 3) tensor in place, top-left corner at pixel (x, y)."""
    color = torch.tensor(color, dtype=torch.uint8)
    for i, ch in enumerate(text.upper()):
        glyph = torch.tensor([c == "#" for c in FONT.get(ch, FONT[" "]).replace(" ", "")]).view(5, 3)
        glyph = glyph.repeat_interleave(scale, 0).repeat_interleave(scale, 1)
        x0 = x + i * 4 * scale
        h, w = glyph.shape
        region = img[y : y + h, x0 : x0 + w]
        region[glyph[: region.shape[0], : region.shape[1]]] = color
    return img


def text_width(text, scale=2):
    return len(text) * 4 * scale - scale


def random_start(n, generator=None, door=None):
    """Uniformly sample n free positions by rejection."""
    pos = torch.empty(n, 2)
    todo = torch.ones(n, dtype=torch.bool)
    while todo.any():
        pos[todo] = torch.rand(int(todo.sum()), 2, generator=generator)
        todo = ~is_free(pos, door)
    return pos


def random_room_point(side, generator=None):
    """side: (N,) bool, True for the left room -> (N, 2) random points clear of the wall and the border."""
    margin = RADIUS + 0.02
    x = margin + torch.rand(len(side), generator=generator) * (WALL_X[0] - RADIUS - 2 * margin)
    x = torch.where(side, x, 1 - x)
    y = margin + torch.rand(len(side), generator=generator) * (1 - 2 * margin)
    return torch.stack([x, y], -1)


def generate(episodes, steps, size=64, seed=0, door_speed=0.0, door_seeking=0.0):
    """Roll out a smooth random-walk policy. Returns a dict of tensors:

    frames  (E, steps + 1, 3, size, size) uint8
    actions (E, steps, 2) float, actions[:, t] moves frames[:, t] to frames[:, t + 1]
    pos     (E, steps + 1, 2) float, ground-truth dot position (only used for evaluation probes)
    door    (E,) float, the door's vertical center, fixed within an episode; with door_speed > 0 it is
            (E, steps + 1) instead: the door starts at a random height and direction and moves door_speed
            per step, bouncing at the ends of its range
    door_seeking: the fraction of episodes (the last ones) whose random walk is pulled toward a waypoint:
    the door, then a random point in the other room, then the door again, which makes many more crossings
    """
    g = torch.Generator().manual_seed(seed)
    # draw doors only when asked, so default datasets stay identical to those made before the option existed
    moving = door_speed > 0
    door = random_door(episodes, g) if moving else None  # None: the exact default DOOR_Y
    direction = torch.where(torch.rand(episodes, generator=g) < 0.5, -1.0, 1.0) if moving else None
    pos = random_start(episodes, g, door)
    action = torch.rand(episodes, 2, generator=g) * 2 - 1
    seeking = torch.arange(episodes) >= episodes - round(door_seeking * episodes)
    if seeking.any():
        to_door = torch.ones(episodes, dtype=torch.bool)  # heading for the door, else for a point in a room
        side = pos[:, 0] < 0.5  # the room the episode started the current leg in
        target = torch.zeros(episodes, 2)
    frames, actions, positions, doors = [render(pos, size, door)], [], [pos], [door]
    for _ in range(steps):
        # a random walk in action space: smooth enough to travel across rooms and through the door
        action = (action + 0.4 * torch.randn(episodes, 2, generator=g)).clamp(-1, 1)
        if seeking.any():
            door_y = door if door is not None else torch.full((episodes,), sum(DOOR_Y) / 2)
            crossed = to_door & ((pos[:, 0] < 0.5) != side)  # through the door: pick a point in this room
            target[crossed] = random_room_point(~side[crossed], g)
            side, to_door = torch.where(crossed, ~side, side), to_door & ~crossed
            arrived = ~to_door & ((pos - target).norm(dim=-1) < 0.1)  # reached it: head for the door again
            to_door |= arrived
            goal = torch.where(to_door[:, None], torch.stack([torch.full_like(door_y, 0.5), door_y], -1), target)
            pull = (goal - pos) / (goal - pos).norm(dim=-1, keepdim=True).clamp_min(1e-6)
            action = torch.where(seeking[:, None], (0.5 * action + 0.8 * pull).clamp(-1, 1), action)
        if moving:
            door, direction = move_door(door, direction, door_speed)
            pos = carry(pos, door)
        pos = step(pos, action, door)
        frames.append(render(pos, size, door))
        actions.append(action)
        positions.append(pos)
        doors.append(door)
    if moving:
        door = torch.stack(doors, 1)
    elif door is None:
        door = torch.full((episodes,), sum(DOOR_Y) / 2)
    return {
        "frames": torch.stack(frames, 1),
        "actions": torch.stack(actions, 1),
        "pos": torch.stack(positions, 1),
        "door": door,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/tworoom.pt")
    p.add_argument("--moving-door", type=float, default=0.0, metavar="SPEED",
                   help="move the door up and down by SPEED per step (e.g. 0.01), from a random start")
    p.add_argument("--door-seeking", type=float, default=0.0, metavar="FRAC",
                   help="fraction of episodes that steer through the door and back instead of a plain random walk")
    args = p.parse_args()

    data = generate(args.episodes, args.steps, args.size, args.seed, args.moving_door, args.door_seeking)
    crossed = ((data["pos"][..., 0] < 0.5).any(1) & (data["pos"][..., 0] > 0.5).any(1)).float().mean()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out)
    mb = data["frames"].numel() / 2**20
    print(f"saved {args.episodes} episodes x {args.steps} steps to {args.out} ({mb:.0f} MB of frames)")
    print(f"{crossed:.0%} of episodes pass through the door")

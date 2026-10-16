"""Train a VisualProbe that decodes Two-Room frames from a frozen LeWM's embeddings.

The probe shows what visual information a latent keeps. LeWM stays frozen: every frame is encoded once,
then the probe learns to map embedding -> frame with an MSE loss in the normalized pixel space.

A probe on the `emb` space (the projector output) also decodes LeWM's imagined futures: starting from one
real frame, `model.rollout` predicts embeddings for the episode's true actions, and the probe draws them.

    python probe.py --space cls                                    # train, save probe.png
    python probe.py --space emb                                    # train, save probe.png + rollout.png
    python probe.py --space emb --load checkpoints/probe_tworoom_6000_moving_slow_seek_emb.pt --rollout-context 3
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import tworoom
from lewm import LeWM, LeWMConfig, VisualProbe
from train import IMAGENET_MEAN, IMAGENET_STD

ZOOM = 2  # frames are drawn at 2x so the labels stay readable
LABEL_SCALE = 2
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (120, 120, 120)


@torch.no_grad()
def embed_all(model, frames, space, mean, std, batch_size=2048):
    """uint8 frames (N, 3, H, W) on device -> (N, D) CLS tokens or projector embeddings."""
    out = []
    for i in range(0, len(frames), batch_size):
        x = (frames[i : i + batch_size].float() / 255 - mean) / std
        z = model.encoder(x)
        out.append(model.projector(z) if space == "emb" else z)
    return torch.cat(out).float()


def dot_position(frames):
    """uint8 frames (N, 3, H, W) -> (N, 2) centroid of the red dot in [0, 1] coords, NaN where none is drawn."""
    r, g, b = frames.int().unbind(1)
    red = (r > 150) & (g < 110) & (b < 110)
    size = frames.size(-1)
    coords = (torch.arange(size, device=frames.device) + 0.5) / size
    count = red.sum((1, 2)).float()
    x = (red * coords).sum((1, 2)) / count
    y = (red * coords[:, None]).sum((1, 2)) / count
    return torch.stack([x, y], 1)  # 0 / 0 = NaN when no red pixel is found


def door_position(frames):
    """uint8 frames (N, 3, H, W) -> (N,) center of the door gap in [0, 1], read from the wall's pixel column.

    Gap rows are those where the background shows through at the wall's x. NaN where no gap is drawn.
    """
    size = frames.size(-1)
    lo, hi = int(tworoom.WALL_X[0] * size) + 1, int(tworoom.WALL_X[1] * size)
    background = (frames[:, 1, :, lo:hi].int() > 150).all(-1)  # (N, H): green is high only on the background
    coords = (torch.arange(size, device=frames.device) + 0.5) / size
    return (background * coords).sum(1) / background.sum(1)  # 0 / 0 = NaN without a gap


def labeled_grid(rows, row_labels, col_labels=None, title=None, gap=4, zoom=ZOOM):
    """rows: list of uint8 (N, 3, S, S) frame rows -> a uint8 (H, W, 3) image with labels on the left and on top."""
    tiles = torch.stack(rows)  # (R, N, 3, S, S)
    R, N = tiles.shape[:2]
    tiles = tiles.repeat_interleave(zoom, -1).repeat_interleave(zoom, -2)
    S = tiles.size(-1)
    char_h = 5 * LABEL_SCALE
    left = max(tworoom.text_width(t, LABEL_SCALE) for t in row_labels) + 3 * gap
    top = (char_h + 2 * gap if col_labels else 0) + (char_h + 3 * gap if title else 0) + gap
    img = torch.full((top + R * (S + gap), left + N * (S + gap), 3), 255, dtype=torch.uint8)
    for r in range(R):
        y = top + r * (S + gap)
        tworoom.draw_text(img, row_labels[r], gap, y + (S - char_h) // 2, BLACK, LABEL_SCALE)
        for n in range(N):
            x = left + n * (S + gap)
            img[y : y + S, x : x + S] = tiles[r, n].permute(1, 2, 0)
    if title:
        tworoom.draw_text(img, title, gap, gap, BLACK, LABEL_SCALE)
    if col_labels:
        y = top - char_h - gap
        for n, text in enumerate(col_labels):
            x = left + n * (S + gap) + (S - tworoom.text_width(text, LABEL_SCALE)) // 2
            tworoom.draw_text(img, text, x, y, GRAY, LABEL_SCALE)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/lewm_tworoom_6000_moving_slow_seek.pt")
    p.add_argument("--data", default="data/tworoom_6000_moving_slow_seek.pt")
    p.add_argument("--space", choices=["cls", "emb"], default="cls",
                   help="cls: the encoder's CLS token, as in the paper; emb: the projector output LeWM plans in")
    p.add_argument("--out", default=None, help="probe checkpoint (default checkpoints/probe_tworoom_6000_moving_slow_seek_<space>.pt)")
    p.add_argument("--load", default=None, help="skip training and use this trained probe checkpoint")
    p.add_argument("--outdir", default=None, help="where the images go (default experiments/probe_6000_moving_slow_seek_<space>)")
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--rollout-steps", type=int, default=12, help="imagined model steps (frameskip raw actions each)")
    p.add_argument("--rollout-episodes", type=int, default=4, help="episodes drawn in rollout.png")
    p.add_argument("--rollout-context", type=int, default=1,
                   help="real frames observed before imagining (more than one shows a moving door's direction)")
    p.add_argument("--val-frac", type=float, default=0.1, help="must match train.py to hold out its val episodes")
    p.add_argument("--seed", type=int, default=0, help="must match train.py's seed for the same split")
    args = p.parse_args()
    args.out = args.out or f"checkpoints/probe_tworoom_6000_moving_slow_seek_{args.space}.pt"
    outdir = Path(args.outdir or f"experiments/probe_6000_moving_slow_seek_{args.space}")
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data = torch.load(args.data)
    num_eps = data["frames"].size(0)

    # the same episode split as train.py: its first random call is this randperm
    perm = torch.randperm(num_eps)
    num_val = max(1, int(num_eps * args.val_frac))
    val_eps, train_eps = perm[:num_val], perm[num_val:]

    ckpt = torch.load(args.ckpt)
    cfg = LeWMConfig(**ckpt["config"])
    model = LeWM(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    for param in model.parameters():
        param.requires_grad_(False)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    to_pixels = lambda frames: (frames.float() / 255 - mean) / std
    to_uint8 = lambda pixels: ((pixels.float() * std + mean).clamp(0, 1) * 255).round().to(torch.uint8)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")

    val_frames = data["frames"][val_eps].flatten(0, 1).to(device)  # uint8 (N, 3, H, W)
    t0 = time.time()
    val_z = embed_all(model, val_frames, args.space, mean, std)

    if args.load:
        state = torch.load(args.load)
        assert state["space"] == args.space, f"{args.load} was trained on the {state['space']} space"
        probe = VisualProbe(in_dim=state["in_dim"], img_size=state["img_size"], patch_size=state["patch_size"])
        probe.load_state_dict(state["probe"])
        probe = probe.to(device).eval()
        print(f"loaded {args.load} ({args.space} space, trained {state['step']} steps)")
    else:
        train_frames = data["frames"][train_eps].flatten(0, 1).to(device)
        train_z = embed_all(model, train_frames, args.space, mean, std)
        print(f"encoded {len(train_z)} train / {len(val_z)} val frames into {args.space} space "
              f"(dim {train_z.size(1)}) in {time.time() - t0:.1f}s")
        probe = train_probe(args, probe_args(train_z, cfg, args), train_z, train_frames, val_z, val_frames,
                            to_pixels, autocast, device)
        del train_frames, train_z
    probe.eval()

    # held-out frames and their reconstructions
    idx = torch.linspace(0, len(val_z) - 1, 8).long().to(device)
    with torch.no_grad(), autocast:
        recon = to_uint8(probe(val_z[idx]))
    img = labeled_grid([val_frames[idx], recon], ["real", "decoded"],
                       title=f"probe on {args.space} space: held-out frames")
    tworoom.save_png(outdir / "probe.png", img.cpu())
    print(f"saved held-out frames and reconstructions to {outdir / 'probe.png'}")

    if args.space == "emb":
        decode_rollouts(args, model, probe, data, val_eps, ckpt, to_pixels, to_uint8, autocast, device, outdir)
    else:
        print("rollouts live in the emb space; train a probe with --space emb to decode them")


def probe_args(train_z, cfg, args):
    return {"in_dim": train_z.size(1), "img_size": cfg.img_size, "patch_size": args.patch_size}


def train_probe(args, shape, train_z, train_frames, val_z, val_frames, to_pixels, autocast, device):
    probe = VisualProbe(**shape).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-3)
    schedule = lambda s: (s + 1) / args.warmup if s < args.warmup else 0.5 * (
        1 + math.cos(math.pi * (s - args.warmup) / max(1, args.steps - args.warmup))
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

    # baseline: predict the mean training frame, i.e. a probe that learned nothing from the embedding
    mean_frame = torch.stack([to_pixels(train_frames[i : i + 4096]).sum(0) for i in range(0, len(train_frames), 4096)])
    mean_frame = mean_frame.sum(0) / len(train_frames)

    @torch.no_grad()
    def evaluate():
        probe.eval()
        mse = base = 0.0
        for i in range(0, len(val_z), 2048):
            target = to_pixels(val_frames[i : i + 2048])
            with autocast:
                recon = probe(val_z[i : i + 2048])
            mse += F.mse_loss(recon.float(), target, reduction="sum").item()
            base += (mean_frame - target).square().sum().item()
        probe.train()
        n = val_frames[0].numel() * len(val_frames)
        return mse / n, base / n

    num_params = sum(p.numel() for p in probe.parameters()) / 1e6
    print(f"device={device} probe params={num_params:.2f}M steps={args.steps} batch={args.batch_size}")
    t0, running = time.time(), 0.0
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(train_z), (args.batch_size,), device=device)
        with autocast:
            loss = F.mse_loss(probe(train_z[idx]).float(), to_pixels(train_frames[idx]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()

        if step % args.eval_every == 0 or step == args.steps:
            val_mse, base_mse = evaluate()
            print(f"step {step:6d} | {time.time() - t0:5.1f}s | lr {sched.get_last_lr()[0]:.1e} | "
                  f"train mse {running / args.eval_every:.4f} | val mse {val_mse:.4f} vs mean frame {base_mse:.4f}",
                  flush=True)
            running = 0.0
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"probe": probe.state_dict(), **shape, "space": args.space, "lewm_ckpt": args.ckpt,
                        "step": step}, args.out)
    return probe


@torch.no_grad()
def decode_rollouts(args, model, probe, data, val_eps, ckpt, to_pixels, to_uint8, autocast, device, outdir):
    """Imagine each held-out episode from its first frames and its true actions, then decode every step.

    Model step k is raw frame k * frameskip; its action is the next frameskip raw actions, normalized as in
    training. The first `--rollout-context` frames are observed, the rest is imagined, as when planning.
    """
    fs, K = ckpt["frameskip"], args.rollout_steps
    c = max(1, min(args.rollout_context, model.config.history_size))
    assert K * fs <= data["actions"].size(1), f"episodes have {data['actions'].size(1)} raw steps, need {K * fs}"
    frame_idx = torch.arange(K + 1) * fs
    real = data["frames"][val_eps][:, frame_idx].to(device)  # (E, K + 1, 3, H, W)
    true_pos = data["pos"][val_eps][:, frame_idx].to(device)  # (E, K + 1, 2)
    actions = data["actions"][val_eps][:, : K * fs].to(device)
    actions = ((actions - ckpt["action_mean"].to(device)) / ckpt["action_std"].to(device)).view(-1, K, fs * 2)

    imagined = []
    for i in range(0, len(real), 256):
        with autocast:
            emb = model.rollout(to_pixels(real[i : i + 256, :c]), actions[i : i + 256])  # (B, K + 1, D)
            imagined.append(to_uint8(probe(emb.float())))
    imagined = torch.cat(imagined)  # (E, K + 1, 3, H, W)

    # how far the imagined dot drifts from the real one, read off the decoded frames
    E = len(real)
    pos = dot_position(imagined.flatten(0, 1)).view(E, K + 1, 2)
    err = (pos - true_pos).norm(dim=-1)  # NaN where the decoded frame has no dot
    stay = (true_pos[:, :1] - true_pos).norm(dim=-1)  # baseline: the dot never moves
    print(f"rollouts on {E} held-out episodes, decoded dot position vs. the real one (arena is 1 x 1):")
    print("  model step   " + " ".join(f"{k:>6d}" for k in range(K + 1)))
    print("  imagined     " + " ".join(f"{e:6.3f}" for e in err.nanmedian(0).values.tolist()))
    print("  no motion    " + " ".join(f"{e:6.3f}" for e in stay.median(0).values.tolist()))
    print("  dot missing  " + " ".join(f"{m:6.0%}" for m in err.isnan().float().mean(0).tolist()))
    lines = []
    if "door" in data and data["door"].dim() == 2:  # a moving door: does the model imagine it moving?
        true_door = data["door"][val_eps][:, frame_idx].to(device)
        door_err = (door_position(imagined.flatten(0, 1)).view(E, K + 1) - true_door).abs()
        door_stay = (true_door[:, c - 1 : c] - true_door).abs()  # baseline: the door stops where last seen
        lines = [("door imagined", door_err.nanmedian(0).values.tolist(), "{:6.3f}"),
                 ("door stops   ", door_stay.median(0).values.tolist(), "{:6.3f}"),
                 ("door missing ", door_err.isnan().float().mean(0).tolist(), "{:6.0%}")]
        print("decoded door height vs. the real one:")
        for name, vals, fmt in lines:
            print(f"  {name}" + " ".join(fmt.format(v) for v in vals))
    print(f"  (model steps 0-{c - 1} are observed, the rest imagined)")

    # show episodes that move across the wall when there are some, since that is where the model struggles
    side = true_pos[..., 0] < 0.5
    crosses = (side != side[:, :1]).any(1)
    order = torch.cat([crosses.nonzero().squeeze(1), (~crosses).nonzero().squeeze(1)])
    pick = order[: args.rollout_episodes]
    rows, labels = [], []
    for e in pick.tolist():
        rows += [real[e], imagined[e]]
        labels += [f"real {e}", f"imagined {e}"]
    cols = [f"obs {k}" for k in range(c)] + [f"+{k}" for k in range(c, K + 1)]
    img = labeled_grid(rows, labels, cols,
                       title=f"rollouts decoded by the emb probe. {c} observed frames, then +k = model step k "
                             f"({fs}k raw actions), imagined")
    tworoom.save_png(outdir / "rollout.png", img.cpu())
    print(f"saved {len(pick)} real vs. imagined episodes to {outdir / 'rollout.png'}")
    (outdir / "rollout_error.txt").write_text(
        "median distance between decoded and real dot position per model step\n"
        + "step " + " ".join(f"{k:>6d}" for k in range(K + 1)) + "\n"
        + "imag " + " ".join(f"{e:6.3f}" for e in err.nanmedian(0).values.tolist()) + "\n"
        + "stay " + " ".join(f"{e:6.3f}" for e in stay.median(0).values.tolist()) + "\n"
        + "miss " + " ".join(f"{m:6.0%}" for m in err.isnan().float().mean(0).tolist()) + "\n"
        + "".join(f"{name.strip()}: " + " ".join(fmt.format(v) for v in vals) + "\n" for name, vals, fmt in lines)
        + f"model steps 0-{c - 1} observed\n"
    )


if __name__ == "__main__":
    main()

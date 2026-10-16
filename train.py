"""Train LeWM on the offline Two-Room dataset.

    python tworoom.py          # generate data/tworoom.pt once
    python train.py            # train, evaluate every epoch, save checkpoints/lewm_tworoom.pt
"""

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from lewm import LeWM, LeWMConfig, LeWMLoss

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Windows:
    """All (pixels, actions) training windows of T = history_size + num_preds frames, kept on device.

    Frame k of a window is raw frame `start + k * frameskip`, and its action is the `frameskip` raw
    actions taken from that frame on, concatenated, so action_dim = frameskip * raw_action_dim.
    """

    def __init__(self, data, episodes, num_frames, frameskip, action_mean, action_std, device):
        self.frames = data["frames"][episodes].to(device)  # uint8 (E, L + 1, 3, H, W)
        self.actions = ((data["actions"][episodes] - action_mean) / action_std).to(device)  # (E, L, A)
        self.pos = data["pos"][episodes].to(device)  # (E, L + 1, 2), only for the probe
        if "door" in data:  # add the door height as a third probe target, per frame
            door = data["door"][episodes]
            door = door[:, None].expand(-1, self.pos.size(1)) if door.dim() == 1 else door
            self.pos = torch.cat([self.pos, door[..., None].to(device)], -1)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)

        self.frame_offsets = torch.arange(num_frames, device=device) * frameskip  # (T,)
        self.action_offsets = self.frame_offsets[:, None] + torch.arange(frameskip, device=device)  # (T, fs)
        num_eps, num_steps = self.actions.shape[:2]
        num_starts = num_steps - num_frames * frameskip + 1  # the last frame's actions must fit too
        ep, start = torch.meshgrid(torch.arange(num_eps), torch.arange(num_starts), indexing="ij")
        self.index = torch.stack([ep.flatten(), start.flatten()], 1).to(device)

    def __len__(self):
        return len(self.index)

    def get(self, idx):
        ep, start = self.index[idx].unbind(1)
        frame_idx = start[:, None] + self.frame_offsets  # (B, T)
        pixels = (self.frames[ep[:, None], frame_idx].float() / 255 - self.mean) / self.std
        actions = self.actions[ep[:, None, None], start[:, None, None] + self.action_offsets]  # (B, T, fs, A)
        return pixels, actions.flatten(2), self.pos[ep[:, None], frame_idx]

    def batches(self, batch_size, shuffle=True, max_batches=None):
        """Full batches only: SIGReg and BatchNorm both take statistics over the batch."""
        device = self.index.device
        order = torch.randperm(len(self), device=device) if shuffle else torch.arange(len(self), device=device)
        num = len(self) // batch_size if max_batches is None else min(max_batches, len(self) // batch_size)
        for i in range(num):
            yield self.get(order[i * batch_size : (i + 1) * batch_size])


def r2(pred, target):
    """Coefficient of determination per output dim, as a list."""
    sse = (pred - target).square().sum(0)
    sst = (target - target.mean(0)).square().sum(0)
    return torch.where(sst > 1e-9, 1 - sse / sst, torch.nan).tolist()  # NaN for a constant target (fixed door)


@torch.no_grad()
def recalibrate_batchnorm(model, windows, batch_size, num_batches=20):
    """Re-estimate BatchNorm running stats for the current weights.

    The running averages kept during training (momentum 0.1) lag behind weights that are still moving,
    and the eval-mode model is very sensitive to that: val loss can look 10x worse than it is.
    """
    model.eval()
    norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    for bn in norms:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average over the batches below
        bn.train()
    for pixels, actions, _ in windows.batches(batch_size, max_batches=num_batches):
        model(pixels, actions)
    for bn in norms:
        bn.momentum = 0.1
    model.train()


@torch.no_grad()
def evaluate(model, criterion, train, val, batch_size):
    """Validation losses, a no-motion baseline, and a linear probe from embeddings to the dot position."""
    model.eval()
    n = model.config.num_preds

    def collect(windows, shuffle, max_batches):
        out = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0, "copy_loss": 0.0}
        emb_all, pred_all, pos_all = [], [], []
        num_batches = 0
        for pixels, actions, pos in windows.batches(batch_size, shuffle, max_batches):
            emb, pred = model(pixels, actions)
            for k, v in criterion(emb, pred).items():
                out[k] += v.item()
            # baseline: predict that nothing changes, i.e. the next embedding equals the current one
            out["copy_loss"] += (emb[:, :-n] - emb[:, n:]).square().mean().item()
            emb_all.append(emb.cpu()), pred_all.append(pred.cpu()), pos_all.append(pos.cpu())
            num_batches += 1
        out = {k: v / num_batches for k, v in out.items()}
        return out, torch.cat(emb_all), torch.cat(pred_all), torch.cat(pos_all)

    _, emb_tr, _, pos_tr = collect(train, shuffle=True, max_batches=20)
    # shuffled: SIGReg is a statistic over the batch and would flag batches of consecutive windows
    metrics, emb_va, pred_va, pos_va = collect(val, shuffle=True, max_batches=None)

    # fit embedding -> (x, y) on train windows, then score it on val embeddings and on val predictions
    with_bias = lambda x: torch.cat([x.flatten(0, 1), torch.ones(x.size(0) * x.size(1), 1)], 1)
    W = torch.linalg.lstsq(with_bias(emb_tr), pos_tr.flatten(0, 1)).solution
    # per dim: dot x, dot y, and the door height when the data has one; the averages cover the dot only
    metrics["probe_r2_emb_dims"] = r2(with_bias(emb_va) @ W, pos_va.flatten(0, 1))
    metrics["probe_r2_pred_dims"] = r2(with_bias(pred_va) @ W, pos_va[:, n:].flatten(0, 1))
    metrics["probe_r2_emb"] = sum(metrics["probe_r2_emb_dims"][:2]) / 2
    metrics["probe_r2_pred"] = sum(metrics["probe_r2_pred_dims"][:2]) / 2
    model.train()
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tworoom.pt")
    p.add_argument("--out", default="checkpoints/lewm_tworoom.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--warmup", type=float, default=0.05, help="fraction of steps spent on linear warmup")
    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data = torch.load(args.data)
    num_eps, img_size = data["frames"].size(0), data["frames"].size(-1)

    cfg = LeWMConfig(
        action_dim=args.frameskip * data["actions"].size(-1),
        img_size=img_size,
        patch_size=16,  # 64x64 frames -> 16 patches; 3x faster than 8 and enough for one dot
        encoder_depth=6,  # half of the paper's ViT-tiny is plenty for a dot on a plain background
        predictor_depth=4,
    )

    # split by episode so validation windows never share frames with training windows
    perm = torch.randperm(num_eps)
    num_val = max(1, int(num_eps * args.val_frac))
    val_eps, train_eps = perm[:num_val], perm[num_val:]
    action_mean = data["actions"][train_eps].mean((0, 1))
    action_std = data["actions"][train_eps].std((0, 1))
    T = cfg.history_size + cfg.num_preds
    train = Windows(data, train_eps, T, args.frameskip, action_mean, action_std, device)
    val = Windows(data, val_eps, T, args.frameskip, action_mean, action_std, device)

    model = LeWM(cfg).to(device)
    criterion = LeWMLoss(sigreg_weight=args.sigreg_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * (len(train) // args.batch_size)
    warmup_steps = max(1, int(args.warmup * total_steps))
    schedule = lambda s: (s + 1) / warmup_steps if s < warmup_steps else 0.5 * (
        1 + math.cos(math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps))
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={device} params={num_params:.1f}M train_windows={len(train)} val_windows={len(val)} "
          f"steps={total_steps}")

    step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0}
        num_batches = 0
        for pixels, actions, _ in train.batches(args.batch_size):
            with autocast:
                emb, pred = model(pixels, actions)
                losses = criterion(emb, pred)

            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            for k in running:
                running[k] += losses[k].item()
            num_batches += 1
            step += 1

        tr = {k: v / num_batches for k, v in running.items()}
        recalibrate_batchnorm(model, train, args.batch_size)
        ev = evaluate(model, criterion, train, val, args.batch_size)
        print(
            f"epoch {epoch:3d} | {time.time() - t0:5.1f}s | lr {sched.get_last_lr()[0]:.1e} | "
            f"train loss {tr['loss']:.4f} (pred {tr['pred_loss']:.4f}, sigreg {tr['sigreg_loss']:.3f}) | "
            f"val pred {ev['pred_loss']:.4f} vs copy {ev['copy_loss']:.4f} | "
            f"probe R2 emb {ev['probe_r2_emb']:.3f}, pred {ev['probe_r2_pred']:.3f} "
            f"(x/y{'/door' if len(ev['probe_r2_emb_dims']) > 2 else ''} "
            f"{'/'.join(f'{v:.2f}' for v in ev['probe_r2_emb_dims'])})"
        )

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": asdict(cfg),
            "model": model.state_dict(),
            "action_mean": action_mean,
            "action_std": action_std,
            "frameskip": args.frameskip,
            "epoch": epoch,
        }, args.out)


if __name__ == "__main__":
    main()

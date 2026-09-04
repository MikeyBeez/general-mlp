"""Sweep per-layer mixer capacity: how high can each layer's R-squared go, and
does the answer differ by depth? Fits one layer at a time from the frozen
teacher's real inputs, so every cell of the grid is independent."""

import argparse, itertools, json, os, torch
from data import Corpus
from model import GPT, AttentionMLP
from common import (RUNS_DIR, Logger, device_and_dtype, gram_loss, cross_gram_loss,
                    rkd_distance_loss, rkd_angle_loss, r_squared, cosine)
from fit_mixers import load_teacher


@torch.no_grad()
def score(teacher, mixer, layer, corpus, batch, ctx, device, dtype, n_batches=10):
    gen = torch.Generator().manual_seed(4321)
    tot = {"mse": 0.0, "r2": 0.0, "cos": 0.0}
    for _ in range(n_batches):
        x, _ = corpus.batch("val", batch, ctx, device, generator=gen)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            _, _, rec = teacher(x, collect=True)
            pred, _ = mixer(rec[layer]["x_in"])
            tgt = rec[layer]["out"]
            tot["mse"] += (pred.float() - tgt.float()).pow(2).mean().item()
            tot["r2"] += r_squared(pred, tgt)
            tot["cos"] += cosine(pred, tgt)
    return {k: round(v / n_batches, 4) for k, v in tot.items()}


def fit_one(teacher, layer, d_model, ctx, groups, d_hidden, steps, corpus, batch, lr,
            device, dtype, w_gram, w_cross, w_dist, w_angle, log, noise=0.0,
            pool="none", beta=1.0, seed=0):
    torch.manual_seed(seed)
    mixer = AttentionMLP(d_model, ctx, d_hidden=d_hidden, groups=groups,
                         pool=pool, beta=beta).to(device)
    for m in (mixer.mix, mixer.out, mixer.gate):
        GPT._init(m)
    n_params = sum(q.numel() for q in mixer.parameters())
    opt = torch.optim.AdamW(mixer.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(seed)
    for step in range(1, steps + 1):
        x, _ = corpus.batch("train", batch, ctx, device, generator=gen)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            _, _, rec = teacher(x, collect=True)
        x_in, tgt = rec[layer]["x_in"].detach(), rec[layer]["out"].detach()
        if noise:
            x_in = x_in + noise * x_in.std() * torch.randn_like(x_in)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            pred, _ = mixer(x_in)
            loss = (pred.float() - tgt.float()).pow(2).mean()
            if w_gram:
                loss = loss + w_gram * gram_loss(pred, tgt)
            if w_cross:
                loss = loss + w_cross * cross_gram_loss(pred, tgt, x_in)
            if w_dist:
                loss = loss + w_dist * rkd_distance_loss(pred, tgt)
            if w_angle:
                loss = loss + w_angle * rkd_angle_loss(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mixer.parameters(), 1.0)
        opt.step(); sched.step()
    return mixer, n_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="teacher")
    p.add_argument("--run", default="sweep")
    p.add_argument("--layers", default="0,1,2,3,4,5")
    p.add_argument("--groups", default="1,8")
    p.add_argument("--hidden", default="128,256,512,1024")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lrs", default=None, help="comma list of learning rates to sweep instead of --lr")
    p.add_argument("--w_gram", type=float, default=1.0)
    p.add_argument("--w_cross", type=float, default=1.0)
    p.add_argument("--w_dist", type=float, default=0.0)
    p.add_argument("--w_angle", type=float, default=0.0)
    p.add_argument("--pools", default="none")
    p.add_argument("--betas", default="1.0")
    p.add_argument("--noise", default="0", help="comma list: gaussian noise on the mixer input, in units of the input's own std")
    args = p.parse_args()

    device, dtype = device_and_dtype()
    log = Logger(os.path.join(RUNS_DIR, args.run), "sweep")
    corpus = Corpus()
    teacher, targs = load_teacher(args.teacher, device)
    for q in teacher.parameters():
        q.requires_grad_(False)
    ctx, d_model = targs["ctx"], targs["d_model"]
    attn_params = sum(q.numel() for q in teacher.blocks[0].mixer.parameters())
    lr_list = [float(x) for x in args.lrs.split(",")] if args.lrs else [args.lr]
    noise_list = [float(x) for x in args.noise.split(",")]
    pool_list = args.pools.split(",")
    beta_list = [float(x) for x in args.betas.split(",")]
    log.log(event="grid", attn_params_per_layer=attn_params, steps=args.steps,
            layers=args.layers, groups=args.groups, hidden=args.hidden, lrs=lr_list,
            noise=noise_list, w_gram=args.w_gram, w_cross=args.w_cross,
            w_dist=args.w_dist, w_angle=args.w_angle)

    layers = [int(x) for x in args.layers.split(",")]
    groups_list = [int(x) for x in args.groups.split(",")]
    hidden_list = [int(x) for x in args.hidden.split(",")]
    for layer, groups, d_hidden, lr, noise, pool, beta in itertools.product(
            layers, groups_list, hidden_list, lr_list, noise_list, pool_list, beta_list):
        if groups > 1 and (d_hidden % groups or d_model % groups):
            continue
        mixer, n_params = fit_one(teacher, layer, d_model, ctx, groups, d_hidden, args.steps,
                                  corpus, args.batch, lr, device, dtype,
                                  args.w_gram, args.w_cross, args.w_dist, args.w_angle,
                                  log, noise=noise, pool=pool, beta=beta)
        m = score(teacher, mixer, layer, corpus, args.batch, ctx, device, dtype)
        log.log(event="cell", layer=layer, groups=groups, d_hidden=d_hidden, lr=lr, noise=noise,
                pool=pool, beta=beta,
                params=n_params, ratio_vs_attn=round(n_params / attn_params, 2), **m)


if __name__ == "__main__":
    main()

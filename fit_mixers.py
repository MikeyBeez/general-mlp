"""Stage 2: fit one causal context MLP per layer to reproduce that layer's
attention output, from the teacher's real inputs. All layers fit in parallel
from the same frozen teacher pass. Then measure the damage of swapping each
layer alone, and of swapping all layers, before any continued training."""

import argparse
import copy
import os
import torch
from data import Corpus
from model import GPT, make_attention_mlp
from common import (RUNS_DIR, Logger, device_and_dtype, evaluate, bits_per_char,
                    gram_loss, cross_gram_loss, rkd_distance_loss, rkd_angle_loss,
                    readout_alignment_loss, residual_gram_loss, r_squared, cosine)


def load_teacher(run, device):
    ckpt = torch.load(os.path.join(RUNS_DIR, run, "teacher.pt"), map_location=device)
    a = ckpt["args"]
    model = GPT(ckpt["vocab"], a["d_model"], a["layers"], a["heads"], 4 * a["d_model"], a["ctx"], 0.0)
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval(), a


@torch.no_grad()
def layer_metrics(teacher, mixers, corpus, batch, ctx, device, dtype, n_batches=10):
    sums = [{"mse": 0.0, "r2": 0.0, "cos": 0.0} for _ in mixers]
    gen = torch.Generator().manual_seed(4321)
    for _ in range(n_batches):
        x, _ = corpus.batch("val", batch, ctx, device, generator=gen)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            _, _, rec = teacher(x, collect=True)
            for i, m in enumerate(mixers):
                pred, _ = m(rec[i]["x_in"])
                tgt = rec[i]["out"]
                sums[i]["mse"] += (pred.float() - tgt.float()).pow(2).mean().item()
                sums[i]["r2"] += r_squared(pred, tgt)
                sums[i]["cos"] += cosine(pred, tgt)
    return [{k: round(v / n_batches, 4) for k, v in s.items()} for s in sums]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="teacher")
    p.add_argument("--run", default="fit_mse")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--groups", type=int, default=1)
    p.add_argument("--d_hidden", type=int, default=0)
    p.add_argument("--w_gram", type=float, default=0.0)
    p.add_argument("--w_cross", type=float, default=0.0)
    p.add_argument("--w_dist", type=float, default=0.0)
    p.add_argument("--w_angle", type=float, default=0.0)
    p.add_argument("--w_readout", type=float, default=0.0)
    p.add_argument("--w_resid", type=float, default=0.0)
    p.add_argument("--pool", default="none", choices=["none", "concat", "mul"])
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device, dtype = device_and_dtype()
    run_dir = os.path.join(RUNS_DIR, args.run)
    log = Logger(run_dir, "fit")
    corpus = Corpus()
    teacher, targs = load_teacher(args.teacher, device)
    for q in teacher.parameters():
        q.requires_grad_(False)
    ctx = targs["ctx"]

    mixers = torch.nn.ModuleList(
        make_attention_mlp(teacher, i, groups=args.groups, d_hidden=args.d_hidden or None,
                           pool=args.pool, beta=args.beta)
        for i in range(len(teacher.blocks))
    ).to(device)
    for m in mixers:
        GPT._init(m.mix); GPT._init(m.out); GPT._init(m.gate)
    log.log(event="setup", teacher_params=sum(q.numel() for q in teacher.parameters()),
            mixer_params_each=sum(q.numel() for q in mixers[0].parameters()),
            attn_params_each=sum(q.numel() for q in teacher.blocks[0].mixer.parameters()),
            w_gram=args.w_gram, w_cross=args.w_cross, w_dist=args.w_dist,
            w_angle=args.w_angle, w_readout=args.w_readout, w_resid=args.w_resid,
            groups=args.groups, pool=args.pool, beta=args.beta)

    opt = torch.optim.AdamW(mixers.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(args.seed)

    for step in range(1, args.steps + 1):
        x, _ = corpus.batch("train", args.batch, ctx, device, generator=gen)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            _, _, rec = teacher(x, collect=True)
        total = 0.0
        parts = {"mse": 0.0, "gram": 0.0, "cross": 0.0, "dist": 0.0, "angle": 0.0,
                 "readout": 0.0, "resid": 0.0}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            for i, m in enumerate(mixers):
                x_in, tgt = rec[i]["x_in"].detach(), rec[i]["out"].detach()
                attn = rec[i]["attn"]
                pred, _ = m(x_in)
                mse = (pred.float() - tgt.float()).pow(2).mean()
                loss = mse
                parts["mse"] += mse.item()
                if args.w_gram:
                    g = gram_loss(pred, tgt); loss = loss + args.w_gram * g; parts["gram"] += g.item()
                if args.w_cross:
                    c = cross_gram_loss(pred, tgt, x_in); loss = loss + args.w_cross * c; parts["cross"] += c.item()
                if args.w_dist:
                    dd = rkd_distance_loss(pred, tgt); loss = loss + args.w_dist * dd; parts["dist"] += dd.item()
                if args.w_angle:
                    aa = rkd_angle_loss(pred, tgt); loss = loss + args.w_angle * aa; parts["angle"] += aa.item()
                if args.w_readout and attn is not None:
                    ro = readout_alignment_loss(pred, tgt, x_in, attn.detach())
                    loss = loss + args.w_readout * ro; parts["readout"] += ro.item()
                if args.w_resid:
                    rg = residual_gram_loss(pred, tgt, x_in)
                    loss = loss + args.w_resid * rg; parts["resid"] += rg.item()
                total = total + loss
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(mixers.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            log.log(event="fit", step=step, **{k: round(v / len(mixers), 5) for k, v in parts.items()})
        if step % args.eval_every == 0 or step == args.steps:
            log.log(event="layer_metrics", step=step, layers=layer_metrics(teacher, mixers, corpus, args.batch, ctx, device, dtype))

    torch.save({"args": vars(args), "teacher_args": targs, "state": mixers.state_dict()}, os.path.join(run_dir, "mixers.pt"))

    base_val = evaluate(teacher, corpus, "val", args.batch, ctx, device, dtype)
    base_test = evaluate(teacher, corpus, "test", args.batch, ctx, device, dtype)
    log.log(event="teacher_lm", val_nats=round(base_val, 4), dickens_nats=round(base_test, 4))
    for i in range(len(mixers)):
        probe = copy.deepcopy(teacher)
        probe.swap_mixer(i, mixers[i])
        v = evaluate(probe, corpus, "val", args.batch, ctx, device, dtype, n_batches=20)
        log.log(event="swap_one", layer=i, val_nats=round(v, 4), delta=round(v - base_val, 4))
    hybrid = copy.deepcopy(teacher)
    for i in range(len(mixers)):
        hybrid.swap_mixer(i, mixers[i])
    v = evaluate(hybrid, corpus, "val", args.batch, ctx, device, dtype)
    t = evaluate(hybrid, corpus, "test", args.batch, ctx, device, dtype)
    log.log(event="swap_all_untrained", val_nats=round(v, 4), val_bpc=round(bits_per_char(v), 4),
            dickens_nats=round(t, 4), dickens_bpc=round(bits_per_char(t), 4),
            delta_val=round(v - base_val, 4), delta_dickens=round(t - base_test, 4))


if __name__ == "__main__":
    main()

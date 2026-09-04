"""Chained fitting with progressive freezing.

fit_mixers.py fits every layer in parallel on the TEACHER's inputs. That is
convenient and it is wrong in one specific way: at swap time layer i does not
receive the teacher's input, it receives whatever the student stack below it
produced. Those inputs have already drifted. So each mixer is asked at test
time to do a job it was never shown, and the damage compounds with depth --
which is exactly the shape of the failure (single-layer swaps cost 0.045
summed, all-at-once costs 0.101).

This script removes that mismatch. Layers are fitted in order, 0 upward:

  1. Install the already-fitted mixers 0..i-1 in a copy of the teacher and
     FREEZE them. That is the progressive freezing: a converged layer stops
     moving, so the input distribution layer i is learning against is stable.
  2. Push the batch through that hybrid to get x_in as layer i will ACTUALLY
     see it in deployment.
  3. Target = the teacher's own attention module applied to THAT SAME x_in.
     Not the teacher's output on the teacher's input. The target is the
     attention FUNCTION evaluated where the student actually stands.
  4. Fit mixer i to that pair, freeze it, move up.

Step 3 is the whole idea. Attention is a function, not a lookup, so it can be
evaluated at any input -- including inputs the teacher would never itself
produce. That makes a correct target available for free at every layer.
"""

import argparse
import copy
import os
import torch
from data import Corpus
from model import GPT, make_attention_mlp
from common import (RUNS_DIR, Logger, device_and_dtype, evaluate, bits_per_char,
                    r_squared, cosine)
from fit_mixers import load_teacher


def student_input_and_target(teacher, hybrid, x, layer, dtype, device):
    """Return (x_in as the student produces it, attention's answer to it)."""
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype,
                                         enabled=device.type == "cuda"):
        _, _, rec = hybrid(x, collect=True)
        x_in = rec[layer]["x_in"].detach()
        tgt, _ = teacher.blocks[layer].mixer(x_in)
    return x_in, tgt.detach()


@torch.no_grad()
def eval_layer(teacher, hybrid, mixer, corpus, batch, ctx, device, dtype, layer, n_batches=10):
    gen = torch.Generator().manual_seed(4321)
    s = {"mse": 0.0, "r2": 0.0, "cos": 0.0}
    for _ in range(n_batches):
        x, _ = corpus.batch("val", batch, ctx, device, generator=gen)
        x_in, tgt = student_input_and_target(teacher, hybrid, x, layer, dtype, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            pred, _ = mixer(x_in)
        s["mse"] += (pred.float() - tgt.float()).pow(2).mean().item()
        s["r2"] += r_squared(pred, tgt)
        s["cos"] += cosine(pred, tgt)
    return {k: round(v / n_batches, 4) for k, v in s.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="teacher")
    p.add_argument("--run", default="chain")
    p.add_argument("--steps", type=int, default=6000, help="steps PER LAYER")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--d_hidden", type=int, default=512)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device, dtype = device_and_dtype()
    run_dir = os.path.join(RUNS_DIR, args.run)
    log = Logger(run_dir, "chain")
    corpus = Corpus()
    teacher, targs = load_teacher(args.teacher, device)
    for q in teacher.parameters():
        q.requires_grad_(False)
    ctx = targs["ctx"]
    n_layers = len(teacher.blocks)

    base_val = evaluate(teacher, corpus, "val", args.batch, ctx, device, dtype)
    base_test = evaluate(teacher, corpus, "test", args.batch, ctx, device, dtype)
    log.log(event="setup", steps_per_layer=args.steps, groups=args.groups,
            d_hidden=args.d_hidden, layers=n_layers,
            teacher_val=round(base_val, 4), teacher_dickens=round(base_test, 4))

    # hybrid starts as a pure copy of the teacher; each fitted mixer is swapped
    # in as it finishes, so layer i+1 trains against layers 0..i already replaced.
    hybrid = copy.deepcopy(teacher).eval()
    fitted = []

    for layer in range(n_layers):
        mixer = make_attention_mlp(teacher, layer, groups=args.groups,
                                   d_hidden=args.d_hidden or None).to(device)
        GPT._init(mixer.mix); GPT._init(mixer.out); GPT._init(mixer.gate)
        opt = torch.optim.AdamW(mixer.parameters(), lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                    total_steps=args.steps, pct_start=0.1)
        gen = torch.Generator().manual_seed(args.seed + layer)
        for step in range(1, args.steps + 1):
            x, _ = corpus.batch("train", args.batch, ctx, device, generator=gen)
            x_in, tgt = student_input_and_target(teacher, hybrid, x, layer, dtype, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                pred, _ = mixer(x_in)
                loss = (pred.float() - tgt.float()).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mixer.parameters(), 1.0)
            opt.step(); sched.step()
            if step % 500 == 0:
                log.log(event="fit", layer=layer, step=step, mse=round(loss.item(), 6))
            if step % args.eval_every == 0 or step == args.steps:
                log.log(event="layer_metric", layer=layer, step=step,
                        **eval_layer(teacher, hybrid, mixer, corpus, args.batch,
                                     ctx, device, dtype, layer))
        for q in mixer.parameters():          # progressive freezing
            q.requires_grad_(False)
        mixer.eval()
        hybrid.swap_mixer(layer, mixer)       # now layer+1 sees the real thing
        fitted.append(mixer)
        v = evaluate(hybrid, corpus, "val", args.batch, ctx, device, dtype)
        log.log(event="after_layer", layer=layer, swapped=layer + 1,
                val_nats=round(v, 4), delta_val=round(v - base_val, 4),
                mixers=hybrid.mixer_kinds())

    mixers = torch.nn.ModuleList(fitted)
    torch.save({"args": vars(args), "teacher_args": targs, "state": mixers.state_dict()},
               os.path.join(run_dir, "mixers.pt"))
    v = evaluate(hybrid, corpus, "val", args.batch, ctx, device, dtype)
    t = evaluate(hybrid, corpus, "test", args.batch, ctx, device, dtype)
    log.log(event="swap_all_untrained", val_nats=round(v, 4), val_bpc=round(bits_per_char(v), 4),
            dickens_nats=round(t, 4), dickens_bpc=round(bits_per_char(t), 4),
            delta_val=round(v - base_val, 4), delta_dickens=round(t - base_test, 4))


if __name__ == "__main__":
    main()

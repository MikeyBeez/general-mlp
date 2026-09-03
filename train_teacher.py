"""Stage 1: train the attention teacher on Shakespeare + Twain."""

import argparse
import json
import math
import os
import torch
from data import Corpus
from model import GPT
from common import RUNS_DIR, Logger, device_and_dtype, evaluate, bits_per_char


def lr_at(step, total, peak, warmup=100):
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1 + math.cos(math.pi * progress)) + peak * 0.05


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="teacher")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device, dtype = device_and_dtype()
    run_dir = os.path.join(RUNS_DIR, args.run)
    log = Logger(run_dir, "train")
    corpus = Corpus()
    log.log(event="corpus", **corpus.summary())

    model = GPT(corpus.vocab, args.d_model, args.layers, args.heads, 4 * args.d_model, args.ctx, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.log(event="model", params=n_params, mixers=model.mixer_kinds())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    gen = torch.Generator().manual_seed(args.seed)

    best = float("inf")
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, corpus, "val", args.batch, args.ctx, device, dtype)
            test = evaluate(model, corpus, "test", args.batch, args.ctx, device, dtype)
            log.log(event="eval", step=step, val_nats=round(val, 4), val_bpc=round(bits_per_char(val), 4),
                    dickens_nats=round(test, 4), dickens_bpc=round(bits_per_char(test), 4))
            if val < best:
                best = val
                torch.save({"args": vars(args), "vocab": corpus.vocab, "state": model.state_dict(), "step": step,
                            "val_nats": val, "dickens_nats": test}, os.path.join(run_dir, "teacher.pt"))
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr)
        x, y = corpus.batch("train", args.batch, args.ctx, device, generator=gen)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            log.log(event="train", step=step, loss=round(loss.item(), 4), lr=opt.param_groups[0]["lr"])

    model.eval()
    prompt = corpus.encode("ROMEO:\n").unsqueeze(0).to(device)
    sample = corpus.decode(model.generate(prompt, 300)[0].tolist())
    with open(os.path.join(run_dir, "sample.txt"), "w") as f:
        f.write(sample)
    log.log(event="done", best_val_nats=round(best, 4))


if __name__ == "__main__":
    main()

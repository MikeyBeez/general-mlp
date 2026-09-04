"""Stage 3: the scaffold test.

arm A: teacher with every attention sublayer replaced by its fitted MLP, then
       trained further with the plain LM loss. Does learning continue?
arm B: the same all-MLP architecture from random init, same data, same steps.
       Never saw attention. The control.
arm C: the attention teacher trained further for the same steps. The ceiling.
arm D: a BRAND NEW model (random embeddings, MLPs, head) whose mixers are the
       fitted attention copies, FROZEN. Nothing else has ever seen attention.
       Can a network learn language around a fixed, borrowed mixer?
arm E: the same, but the frozen mixers are random. The control that says whether
       arm D's result comes from what the mixer learned or merely from having
       some fixed mixer in the slot.
arm F: arm D's finished model, with the mixers UNLOCKED and everything trained
       together. Does the borrowed part become the model's own?
"""

import argparse
import copy
import os
import torch
from data import Corpus
from model import GPT, make_attention_mlp
from common import RUNS_DIR, Logger, device_and_dtype, evaluate, bits_per_char
from fit_mixers import load_teacher
from train_teacher import lr_at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="teacher")
    p.add_argument("--mixers", default="fit_mse")
    p.add_argument("--arm", choices=["A", "B", "C", "D", "E", "F"], required=True)
    p.add_argument("--init_run", default=None, help="arm F: run whose best.pt to unlock and keep training")
    p.add_argument("--run", default=None)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device, dtype = device_and_dtype()
    run = args.run or f"arm_{args.arm}_{args.mixers}"
    run_dir = os.path.join(RUNS_DIR, run)
    log = Logger(run_dir, "continue")
    corpus = Corpus()
    teacher, targs = load_teacher(args.teacher, device)
    ctx = targs["ctx"]

    if args.arm == "C":
        model = teacher
    else:
        mix_ckpt = torch.load(os.path.join(RUNS_DIR, args.mixers, "mixers.pt"), map_location=device)
        margs = mix_ckpt["args"]
        if args.arm == "A":
            model = teacher
        else:
            model = GPT(corpus.vocab, targs["d_model"], targs["layers"], targs["heads"],
                        4 * targs["d_model"], ctx, 0.0).to(device)
        mixers = torch.nn.ModuleList(
            make_attention_mlp(model, i, groups=margs["groups"], d_hidden=margs["d_hidden"] or None)
            for i in range(len(model.blocks))
        ).to(device)
        if args.arm in ("A", "D", "F"):
            mixers.load_state_dict(mix_ckpt["state"])
        else:
            for m in mixers:
                GPT._init(m.mix); GPT._init(m.out); GPT._init(m.gate)
        for i, m in enumerate(mixers):
            model.swap_mixer(i, m)
    for block in model.blocks:
        block.drop.p = args.dropout
    model.drop.p = args.dropout
    for q in model.parameters():
        q.requires_grad_(True)
    if args.arm == "F":
        ckpt = torch.load(os.path.join(RUNS_DIR, args.init_run, "best.pt"), map_location=device)
        model.load_state_dict(ckpt["state"])
        log.log(event="unlocked_from", run=args.init_run, step=ckpt["step"],
                dickens_nats=round(ckpt["dickens_nats"], 4))
    frozen = 0
    if args.arm in ("D", "E"):
        for block in model.blocks:
            for q in block.mixer.parameters():
                q.requires_grad_(False)
                frozen += q.numel()
    model.train()
    trainable = sum(q.numel() for q in model.parameters() if q.requires_grad)
    log.log(event="setup", arm=args.arm, mixers=model.mixer_kinds(),
            params=sum(q.numel() for q in model.parameters()),
            frozen_params=frozen, trainable_params=trainable)

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    gen = torch.Generator().manual_seed(args.seed + 7)
    best_test = float("inf")
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate(model, corpus, "val", args.batch, ctx, device, dtype)
            t = evaluate(model, corpus, "test", args.batch, ctx, device, dtype)
            log.log(event="eval", step=step, val_nats=round(v, 4), val_bpc=round(bits_per_char(v), 4),
                    dickens_nats=round(t, 4), dickens_bpc=round(bits_per_char(t), 4))
            if t < best_test:
                best_test = t
                torch.save({"arm": args.arm, "state": model.state_dict(), "teacher_args": targs, "step": step,
                            "val_nats": v, "dickens_nats": t}, os.path.join(run_dir, "best.pt"))
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, warmup=50)
        x, y = corpus.batch("train", args.batch, ctx, device, generator=gen)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            log.log(event="train", step=step, loss=round(loss.item(), 4))

    torch.save({"arm": args.arm, "state": model.state_dict(), "teacher_args": targs}, os.path.join(run_dir, "model.pt"))
    model.eval()
    prompt = corpus.encode("It was the best of times, ").unsqueeze(0).to(device)
    with open(os.path.join(run_dir, "sample.txt"), "w") as f:
        f.write(corpus.decode(model.generate(prompt, 300)[0].tolist()))
    log.log(event="done", arm=args.arm, best_dickens_nats=round(best_test, 4))


if __name__ == "__main__":
    main()

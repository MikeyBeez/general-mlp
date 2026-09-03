import json
import os
import time
import torch
import torch.nn.functional as F

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("GMLP_RUNS", "runs"))


def device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


@torch.no_grad()
def evaluate(model, corpus, split, batch_size, ctx, device, dtype, n_batches=40, seed=1234):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(n_batches):
        x, y = corpus.batch(split, batch_size, ctx, device, generator=gen)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            _, loss, _ = model(x, y)
        total += loss.item()
    model.train()
    return total / n_batches


def bits_per_char(nats):
    return nats / 0.6931471805599453


class Logger:
    def __init__(self, run_dir, name):
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, name + ".jsonl")
        self.t0 = time.time()

    def log(self, **record):
        record["elapsed_s"] = round(time.time() - self.t0, 1)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def gram(h):
    h = F.normalize(h.float(), dim=-1)
    return h @ h.transpose(1, 2)


def gram_loss(h_student, h_teacher):
    return (gram(h_student) - gram(h_teacher)).pow(2).mean()


def cross_gram_loss(out_student, out_teacher, x_in):
    """Pair geometry between each output and every input position: this is
    where attention's score table lives, so matching it keeps the pairs."""
    x = F.normalize(x_in.float(), dim=-1)
    cs = F.normalize(out_student.float(), dim=-1) @ x.transpose(1, 2)
    ct = F.normalize(out_teacher.float(), dim=-1) @ x.transpose(1, 2)
    return (cs - ct).pow(2).mean()


def r_squared(pred, target):
    pred, target = pred.float(), target.float()
    ss_res = (pred - target).pow(2).sum()
    ss_tot = (target - target.mean()).pow(2).sum()
    return (1 - ss_res / ss_tot).item()


def cosine(pred, target):
    return F.cosine_similarity(pred.float(), target.float(), dim=-1).mean().item()

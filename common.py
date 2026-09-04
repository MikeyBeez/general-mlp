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


def rkd_distance_loss(pred, target):
    """Relational KD (Park et al. 2019): match the pairwise distances between
    token outputs, each set normalized by its own mean so overall scale cancels
    and only the relative geometry has to agree."""
    d_p = torch.cdist(pred.float(), pred.float())
    d_t = torch.cdist(target.float(), target.float())
    d_p = d_p / d_p.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    d_t = d_t / d_t.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return F.smooth_l1_loss(d_p, d_t)


def _gather_rows(h, idx):
    return h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.size(-1)))


def rkd_angle_loss(pred, target, n_triplets=256):
    """Match the angle at anchor a between (p - a) and (q - a) for random
    triplets of positions: a three-body invariant the distance term misses."""
    b, n, _ = pred.shape
    idx = torch.randint(0, n, (b, n_triplets, 3), device=pred.device)

    def angles(h):
        h = h.float()
        a, p, q = (_gather_rows(h, idx[..., i]) for i in range(3))
        return (F.normalize(p - a, dim=-1) * F.normalize(q - a, dim=-1)).sum(dim=-1)

    return F.smooth_l1_loss(angles(pred), angles(target))


def attention_profile_loss(pred, x_in, attn_weights, tau=10.0, eps=1e-8):
    """Match WHERE the output reads from, using the teacher's own score table
    as the target.

    The reasoning: attention's output at position t is a weighted sum of values
    at positions j, with weights a[t, j]. Those weights ARE the thing attention
    computes and the MLP must not compute. But they are available at fit time,
    from the teacher, for free.

    So instead of asking the student to match the teacher's output vector (which
    is what MSE does) or the geometry of its outputs among themselves (which is
    what the Gram terms do), ask a sharper question: does the student's output
    at t lean on the same input positions the teacher leaned on?

    The student's lean on position j is measured as the cosine between its
    output at t and the input at j. Softmax that row and you have a
    distribution over positions, directly comparable to the teacher's attention
    row. Cross-entropy between the two is the loss.

    This is the only term here that uses the score matrix itself. It is the
    scaffold made explicit: the Cartesian product supervises training and is
    absent at inference.
    """
    x = F.normalize(x_in.float(), dim=-1)
    p = F.normalize(pred.float(), dim=-1)
    # tau is not decoration: cosines live in [-1, 1], so an untempered softmax
    # over them is nearly uniform and CANNOT express a peaked attention row.
    # Measured: raising tau from 1 to 50 widened the gap between a teacher-like
    # prediction and a random one from 2% of the loss to 7%.
    logits = (p @ x.transpose(1, 2)) * tau               # (b, n, n) lean scores
    n = logits.size(1)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=logits.device), 1)
    logits = logits.masked_fill(mask, float("-inf"))
    log_student = F.log_softmax(logits, dim=-1).masked_fill(mask, 0.0)
    teacher = attn_weights.float()
    if teacher.dim() == 4:                               # (b, heads, n, n)
        teacher = teacher.mean(dim=1)                    # heads agree on where, not how much
    # The teacher must be zeroed on the SAME causal mask before normalizing.
    # Leaving mass on masked positions multiplies 0 by -inf and the loss is inf
    # rather than large -- a silent poison, not a visible error.
    teacher = teacher.masked_fill(mask, 0.0)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(teacher * log_student).sum(dim=-1).mean()


def readout_alignment_loss(pred, target, x_in, attn_weights):
    """Keep the same angular relationship to WHERE THE TEACHER LOOKED.

    Build the teacher's readout r_t = sum_j a[t, j] * x_j: the teacher's own
    attention row applied to this layer's input. That is a pure statement of
    where attention read, with the value and output projections left out of it.

    Then require one number per position to agree: the cosine between the
    output and that readout. The student is not asked to BE the readout - it
    still has to produce the projected, transformed vector - only to stand in
    the same direction relative to it as the teacher did.

    Chosen over a distribution-matching loss on the score table after measuring
    both: this one is exactly zero on the teacher's own output, rises
    monotonically with damage (0.0003 / 0.003 / 0.010 for 0.1 / 0.3 / 1.0 noise)
    and is cheap. The score-table version was nearly flat by comparison.
    """
    a = attn_weights.float()
    if a.dim() == 4:
        a = a.mean(dim=1)
    r = a @ x_in.float()
    cp = F.cosine_similarity(pred.float(), r, dim=-1)
    ct = F.cosine_similarity(target.float(), r, dim=-1)
    return F.smooth_l1_loss(cp, ct)


def residual_gram_loss(pred, target, x_in):
    """The Gram terms above compare the sublayer's OUTPUT. That is the wrong
    object. The block computes x + mixer(norm(x)), so the mixer's output is a
    small correction, and what the next layer actually sees - and what errors
    compound through - is the residual STREAM. Two corrections can have very
    different internal geometry and still leave the stream almost identical,
    and vice versa. So compare the streams."""
    return gram_loss(x_in + pred, x_in + target)


def r_squared(pred, target):
    pred, target = pred.float(), target.float()
    ss_res = (pred - target).pow(2).sum()
    ss_tot = (target - target.mean()).pow(2).sum()
    return (1 - ss_res / ss_tot).item()


def cosine(pred, target):
    return F.cosine_similarity(pred.float(), target.float(), dim=-1).mean().item()

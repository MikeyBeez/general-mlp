"""GPT teacher whose attention sublayers can be swapped for causal context MLPs.

Every block exposes `mixer`, which is either `CausalSelfAttention` or
`AttentionMLP`. Both map a normalized (b, n, d) input to a (b, n, d) output,
so the rest of the block does not care which one is installed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, return_weights=False):
        b, n, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q = q.view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        if return_weights:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
            scores = scores.masked_fill(mask, float("-inf"))
            weights = scores.softmax(dim=-1)
            out = weights @ v
        else:
            weights = None
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out), weights


class AttentionMLP(nn.Module):
    """Causal context MLP that stands in for attention.

    Position t reads the last `window` normalized inputs through one shared
    weight (a Conv1d with kernel `window` is exactly that), passes the result
    through a nonlinearity and a linear layer, and gates it with the current
    token. `groups` > 1 restricts which channels may mix across positions.
    """

    def __init__(self, d_model, window, d_hidden=None, groups=1):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.window = window
        self.mix = nn.Conv1d(d_model, d_hidden, kernel_size=window, groups=groups)
        self.out = nn.Linear(d_hidden, d_model)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x, return_weights=False):
        padded = F.pad(x.transpose(1, 2), (self.window - 1, 0))
        h = F.gelu(self.mix(padded).transpose(1, 2))
        return self.out(h) * torch.sigmoid(self.gate(x)), None


class MLP(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.up = nn.Linear(d_model, d_hidden)
        self.down = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, collect=None):
        h = self.norm1(x)
        mixed, weights = self.mixer(h, return_weights=collect is not None)
        if collect is not None:
            collect.append({"x_in": h, "out": mixed, "attn": weights})
        x = x + self.drop(mixed)
        return x + self.drop(self.mlp(self.norm2(x)))


class GPT(nn.Module):
    def __init__(self, vocab, d_model=256, n_layers=6, n_heads=8, d_hidden=1024,
                 ctx=256, dropout=0.0):
        super().__init__()
        self.ctx = ctx
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(ctx, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_hidden, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.drop = nn.Dropout(dropout)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, targets=None, collect=False):
        n = tokens.size(1)
        x = self.embed(tokens) + self.pos(torch.arange(n, device=tokens.device))
        x = self.drop(x)
        records = [] if collect else None
        for block in self.blocks:
            x = block(x, collect=records)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss, records

    def mixer_kinds(self):
        return [type(b.mixer).__name__ for b in self.blocks]

    def swap_mixer(self, layer, mixer):
        self.blocks[layer].mixer = mixer

    @torch.no_grad()
    def generate(self, tokens, n_new, temperature=1.0):
        for _ in range(n_new):
            logits, _, _ = self(tokens[:, -self.ctx:])
            probs = (logits[:, -1] / temperature).softmax(dim=-1)
            tokens = torch.cat([tokens, torch.multinomial(probs, 1)], dim=1)
        return tokens


def make_attention_mlp(model, layer, groups=1, d_hidden=None):
    d_model = model.embed.weight.size(1)
    return AttentionMLP(d_model, model.ctx, d_hidden=d_hidden, groups=groups)

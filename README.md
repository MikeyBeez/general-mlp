# General MLP

Can a plain MLP learn what an attention layer does — well enough that you can
freeze the copy, build a brand new network around it, and have that network
learn language?

Short answer, on this setup: yes.

## The experiment

1. **Train a teacher.** A small character-level GPT (6 layers, d=256, 8 heads,
   256-char context) on Shakespeare + Twain. Held-out author: Dickens.
2. **Copy each attention layer.** For every layer, record what that layer's
   attention sublayer saw and what it produced, then fit an `AttentionMLP` — a
   causal convolution over the whole context, a nonlinearity, a linear layer,
   and a gate — to reproduce the mapping. Optional structural regularizers match
   the *pair geometry* of the outputs (a Gram matrix of cosine similarities, and
   the output-to-input cosine table where attention's score matrix lives).
3. **Swap and continue**, or **freeze and rebuild**, and see what happens.

## Arms

| arm | what it is |
| --- | --- |
| A | teacher with all 6 attention sublayers replaced by fitted MLPs, trained further |
| B | same architecture from scratch, mixers random and trainable — never saw attention |
| C | the attention teacher, trained further — the ceiling |
| D | **brand new model, fitted MLPs frozen in**, everything else random |
| E | same as D but the frozen mixers are random — the control for D |

## Headline numbers

Dickens (held-out author), nats per character, lower is better. Teacher: 1.262.

- D (frozen fitted mixers, 3.25M trainable params): **1.326**
- B (from scratch, all 16.6M params trainable): 1.363
- E (frozen *random* mixers): 1.550
- A (swapped teacher, best over all schedules): 1.324

D beats B while training a fifth as many parameters, and beats E by 0.22 — so
it is what the copy learned, not merely having something fixed in the slot.

Per-layer copy quality (R² on held-out text, layers 0→5, 20k-step fit):
0.98, 0.90, 0.85, 0.75, 0.68, 0.61. Early layers are nearly free to replace;
the cost grows with depth.

## Running it

```bash
./get_data.sh
export GMLP_CORPUS=big GMLP_RUNS=runs_big
python train_teacher.py --run teacher --steps 12000
python fit_mixers.py --run fit_g8 --steps 20000 --groups 8 --w_gram 1 --w_cross 1
python continue_train.py --arm D --mixers fit_g8 --run arm_D --steps 8000 --lr 1e-3
python continue_train.py --arm E --mixers fit_g8 --run arm_E --steps 8000 --lr 1e-3
```

Needs PyTorch with CUDA. The whole pipeline is minutes-to-an-hour on a single
16GB card. `logs/` holds the raw JSONL from every run reported above.

## Files

- `data.py` — character corpus, train/test split by author
- `model.py` — GPT, `CausalSelfAttention`, `AttentionMLP`, hot-swappable mixer slot
- `fit_mixers.py` — stage 2: fit the copies, measure them, measure the swap cost
- `continue_train.py` — stage 3: arms A–E
- `common.py` — losses (`gram_loss`, `cross_gram_loss`), eval, logging

## Prior work

This continues a line of the author's earlier experiments on approximating
attention with feedforward networks (2024–2025), which established that single
heads and whole deep layers can be replaced with small MLPs at near-parity. What
is new here is the structural regularizer, the held-out-author test, and arm D:
freezing the copy and training a new model around it.

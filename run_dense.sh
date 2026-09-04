cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
set -x

# The config every overnight sweep should have used. DENSE mixer (groups=1,
# 16.9M params/layer), 40k steps -- double the best-known run (big_fit_long,
# dense + gram/cross at 20k, dickens 1.3462). Two arms, matched steps, both
# fully annealed, so the regularizer question is answered cleanly at density.

$P fit_mixers.py --run d_plain --steps 40000 --groups 1 --d_hidden 0 \
   --eval_every 10000 2>&1 | tee logs/d_plain.log

$P fit_mixers.py --run d_gramcross --steps 40000 --groups 1 --d_hidden 0 \
   --eval_every 10000 --w_gram 1 --w_cross 1 2>&1 | tee logs/d_gramcross.log

echo DENSE_DONE

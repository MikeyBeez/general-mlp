cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
set -x

# 1. CHAINED FITTING + PROGRESSIVE FREEZING. Each layer fitted on the input the
#    student actually produces, targeting attention evaluated at that same input.
$P chain_fit.py --run n_chain --steps 10000 --groups 8 --d_hidden 512 \
   --eval_every 2500 2>&1 | tee logs/n_chain.log

# 2. The steps lever, clean. Regularizer ablation says plain MSE; the 20k run
#    says more steps is the only thing that has ever moved the deep layers.
$P fit_mixers.py --run n_long_clean --steps 60000 --groups 8 --d_hidden 512 \
   --eval_every 5000 2>&1 | tee logs/n_long_clean.log

echo QUEUE2_DONE

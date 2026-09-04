cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python

# wait for queue 2 to finish rather than fighting it for the card
while ! grep -q QUEUE2_DONE /tmp/night2.out 2>/dev/null; do sleep 30; done
set -x

# 1. THE PAYOFF TEST. Arm D with the CHAINED mixers: brand new model, chained
#    mixers frozen in, nothing else has ever seen attention. Beat 1.3261?
$P continue_train.py --arm D --mixers n_chain --run arm_D_chain --steps 6000 \
   --lr 1e-3 --eval_every 250 2>&1 | tee logs/arm_D_chain.log

# 2. Steps lever applied to the method that works.
$P chain_fit.py --run n_chain_long --steps 30000 --groups 8 --d_hidden 512 \
   --eval_every 10000 2>&1 | tee logs/n_chain_long.log

echo QUEUE3_DONE

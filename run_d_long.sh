#!/bin/bash
cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
$P continue_train.py --arm D --mixers fit_g8 --run arm_D_hi_long --steps 24000 --lr 1e-3 --dropout 0.1 --eval_every 1000 2>&1 | tee big_arm_D_hi_long.log
$P continue_train.py --arm B --mixers fit_g8 --run arm_B_hi_long --steps 24000 --lr 1e-3 --dropout 0.1 --eval_every 1000 2>&1 | tee big_arm_B_hi_long.log
$P continue_train.py --arm E --mixers fit_g8 --run arm_E_hi_long --steps 24000 --lr 1e-3 --dropout 0.1 --eval_every 1000 2>&1 | tee big_arm_E_hi_long.log

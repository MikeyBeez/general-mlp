#!/bin/bash
cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
$P fit_mixers.py --run fit_long --steps 20000 --eval_every 2500 --w_gram 1 --w_cross 1 2>&1 | tee big_fit_long.log
$P continue_train.py --arm A --mixers fit_long --run arm_A_long --steps 12000 --eval_every 500 2>&1 | tee big_arm_A_long.log
$P continue_train.py --arm B --mixers fit_long --run arm_B_long --steps 12000 --eval_every 500 2>&1 | tee big_arm_B_long.log

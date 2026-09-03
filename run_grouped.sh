#!/bin/bash
cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
$P fit_mixers.py --run fit_g8 --steps 20000 --eval_every 2500 --groups 8 --w_gram 1 --w_cross 1 2>&1 | tee big_fit_g8.log
$P continue_train.py --arm A --mixers fit_g8 --run arm_A_g8 --steps 6000 --eval_every 250 2>&1 | tee big_arm_A_g8.log
$P continue_train.py --arm B --mixers fit_g8 --run arm_B_g8 --steps 6000 --eval_every 250 2>&1 | tee big_arm_B_g8.log

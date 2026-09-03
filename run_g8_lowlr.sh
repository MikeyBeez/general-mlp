#!/bin/bash
cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
$P continue_train.py --arm A --mixers fit_g8 --run arm_A_g8_lowlr --steps 8000 --lr 1e-4 --dropout 0.2 --eval_every 500 2>&1 | tee big_arm_A_g8_lowlr.log
$P continue_train.py --arm B --mixers fit_g8 --run arm_B_g8_lowlr --steps 8000 --lr 1e-4 --dropout 0.2 --eval_every 500 2>&1 | tee big_arm_B_g8_lowlr.log

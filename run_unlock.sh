#!/bin/bash
cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
$P continue_train.py --arm F --mixers fit_g8 --init_run arm_D_hi --run arm_F_unlock --steps 6000 --lr 3e-4 --dropout 0.1 --eval_every 250 2>&1 | tee logs/big_arm_F_unlock.log
$P continue_train.py --arm F --mixers fit_g8 --init_run arm_D_hi --run arm_F_unlock_gentle --steps 6000 --lr 1e-4 --dropout 0.15 --eval_every 250 2>&1 | tee logs/big_arm_F_unlock_gentle.log

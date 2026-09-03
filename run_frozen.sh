#!/bin/bash
cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
# matched to arm B_g8_lowlr (lr 1e-4): fresh model, mixers frozen
$P continue_train.py --arm D --mixers fit_g8 --run arm_D_lowlr --steps 8000 --lr 1e-4 --dropout 0.2 --eval_every 500 2>&1 | tee big_arm_D_lowlr.log
$P continue_train.py --arm E --mixers fit_g8 --run arm_E_lowlr --steps 8000 --lr 1e-4 --dropout 0.2 --eval_every 500 2>&1 | tee big_arm_E_lowlr.log
# a fresh model deserves a real learning rate too: full trio at 1e-3
$P continue_train.py --arm D --mixers fit_g8 --run arm_D_hi --steps 8000 --lr 1e-3 --dropout 0.1 --eval_every 500 2>&1 | tee big_arm_D_hi.log
$P continue_train.py --arm E --mixers fit_g8 --run arm_E_hi --steps 8000 --lr 1e-3 --dropout 0.1 --eval_every 500 2>&1 | tee big_arm_E_hi.log
$P continue_train.py --arm B --mixers fit_g8 --run arm_B_hi --steps 8000 --lr 1e-3 --dropout 0.1 --eval_every 500 2>&1 | tee big_arm_B_hi.log

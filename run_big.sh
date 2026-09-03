#!/bin/bash
cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
$P train_teacher.py --run teacher --steps 12000 --eval_every 500 2>&1 | tee big_teacher.log
$P fit_mixers.py --run fit_mse --steps 4000 2>&1 | tee big_fit_mse.log
$P fit_mixers.py --run fit_both --steps 4000 --w_gram 1 --w_cross 1 2>&1 | tee big_fit_both.log
$P continue_train.py --arm C --run arm_C --steps 4000 2>&1 | tee big_arm_C.log
$P continue_train.py --arm A --mixers fit_mse --steps 4000 2>&1 | tee big_arm_A_mse.log
$P continue_train.py --arm A --mixers fit_both --steps 4000 2>&1 | tee big_arm_A_both.log
$P continue_train.py --arm B --mixers fit_mse --run arm_B --steps 4000 2>&1 | tee big_arm_B.log

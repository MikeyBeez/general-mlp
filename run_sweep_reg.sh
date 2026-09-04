#!/bin/bash
cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
L="0,3,5"
C="--layers $L --groups 8 --hidden 512 --steps 4000"
# 1. does noise on the mixer input stop it memorising the teacher?
$P sweep_layers.py --run sw_noise $C --noise 0,0.02,0.05,0.1,0.2 2>&1 | tee logs/big_sw_noise.log
# 2. regularizer ablation: what is each structural term actually worth?
$P sweep_layers.py --run sw_reg_none  $C --w_gram 0 --w_cross 0 2>&1 | tee logs/big_sw_reg_none.log
$P sweep_layers.py --run sw_reg_gram  $C --w_gram 1 --w_cross 0 2>&1 | tee logs/big_sw_reg_gram.log
$P sweep_layers.py --run sw_reg_cross $C --w_gram 0 --w_cross 1 2>&1 | tee logs/big_sw_reg_cross.log
$P sweep_layers.py --run sw_reg_gc    $C --w_gram 1 --w_cross 1 2>&1 | tee logs/big_sw_reg_gc.log
$P sweep_layers.py --run sw_reg_rkd   $C --w_gram 0 --w_cross 0 --w_dist 1 --w_angle 1 2>&1 | tee logs/big_sw_reg_rkd.log
$P sweep_layers.py --run sw_reg_all   $C --w_gram 1 --w_cross 1 --w_dist 1 --w_angle 1 2>&1 | tee logs/big_sw_reg_all.log

#!/bin/bash
cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
# learning-rate sweep, deep layers vs a shallow reference, at the best-so-far shape
$P sweep_layers.py --run sweep_lr --layers 0,3,4,5 --groups 8 --hidden 512 \
  --lrs 3e-4,1e-3,3e-3,1e-2 --steps 4000 2>&1 | tee logs/big_sweep_lr.log
# and a longer run at the winning lr band for the hardest layer
$P sweep_layers.py --run sweep_lr_long --layers 4,5 --groups 8 --hidden 512 \
  --lrs 1e-3,3e-3 --steps 16000 2>&1 | tee logs/big_sweep_lr_long.log

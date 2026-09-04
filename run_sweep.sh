#!/bin/bash
cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
$P sweep_layers.py --run sweep_cap --layers 0,1,2,3,4,5 --groups 1,8 --hidden 128,256,512,1024 --steps 4000 2>&1 | tee logs/big_sweep_cap.log

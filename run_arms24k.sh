cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
set -x

# Rebuild the article headline on DENSE mixers at 24,000 steps -- the same
# budget the published narrow-mixer arms used (D 1.3035, B 1.3717, E 1.4513).
# The 6,000-step version was still improving, so it could not be compared.
$P continue_train.py --arm D --mixers d_gramcross --run aD_dense24 --steps 24000 \
   --lr 1e-3 --eval_every 1000 2>&1 | tee logs/aD_dense24.log
$P continue_train.py --arm E --mixers d_gramcross --run aE_dense24 --steps 24000 \
   --lr 1e-3 --eval_every 1000 2>&1 | tee logs/aE_dense24.log
$P continue_train.py --arm B --mixers d_gramcross --run aB_dense24 --steps 24000 \
   --lr 1e-3 --eval_every 1000 2>&1 | tee logs/aB_dense24.log
echo ARMS24_DONE

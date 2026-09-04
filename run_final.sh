cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
set -x

# STAGE 1 - rebuild the article headline on the DENSE mixers. D is the claim;
# E (frozen random) and B (all trainable) are its controls, rerun so the three
# numbers come from one consistent pipeline.
$P continue_train.py --arm D --mixers d_gramcross --run fD_dense --steps 6000 \
   --lr 1e-3 --eval_every 250 2>&1 | tee logs/fD_dense.log
$P continue_train.py --arm E --mixers d_gramcross --run fE_dense --steps 6000 \
   --lr 1e-3 --eval_every 250 2>&1 | tee logs/fE_dense.log
$P continue_train.py --arm B --mixers d_gramcross --run fB_dense --steps 6000 \
   --lr 1e-3 --eval_every 250 2>&1 | tee logs/fB_dense.log
echo STAGE1_DONE

# STAGE 2 - chained fitting at density. Never run; chaining was only ever
# tested on the narrow mixer where the comparison was inconclusive.
$P chain_fit.py --run fchain_dense --steps 10000 --groups 1 --d_hidden 0 \
   --eval_every 5000 2>&1 | tee logs/fchain_dense.log
echo STAGE2_DONE

# STAGE 3 - dense past 40k. Layer 5 was still climbing at 0.6441.
$P fit_mixers.py --run fd_120k --steps 120000 --groups 1 --d_hidden 0 \
   --eval_every 20000 --w_gram 1 --w_cross 1 2>&1 | tee logs/fd_120k.log
echo STAGE3_DONE

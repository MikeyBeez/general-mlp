cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
while ! grep -q QUEUE3_DONE /tmp/night3.out 2>/dev/null; do sleep 30; done
set -x

# Steps is the ONLY lever that has moved the deep layers, and layer 5 was still
# climbing at 60k with no sign of a plateau. Run it four times longer and find
# out where it actually stops.
$P fit_mixers.py --run n_vlong_clean --steps 250000 --groups 8 --d_hidden 512 \
   --eval_every 10000 2>&1 | tee logs/n_vlong_clean.log

echo QUEUE4_DONE

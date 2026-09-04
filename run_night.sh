#!/bin/bash
cd ~/Code/general-mlp
export GMLP_CORPUS=big GMLP_RUNS=runs_big
P=~/Code/HRS/.venv/bin/python
set -x

# 1. ENGRAM + GATE SHARPNESS, the two ideas Mikey proposed. Layers 0/3/5.
$P sweep_layers.py --run n_engram --layers 0,3,5 --groups 8 --hidden 512 --steps 4000 \
   --pools none,concat,mul --betas 1.0,2.0 2>&1 | tee logs/n_engram.log

# 2. Same, at the settings that won the capacity sweep for the DEEP layers (dense).
$P sweep_layers.py --run n_engram_dense --layers 3,5 --groups 1 --hidden 1024 --steps 4000 \
   --pools none,concat,mul 2>&1 | tee logs/n_engram_dense.log

# 3. REGULARIZER ABLATION SCORED ON SWAP-ALL (the metric that actually matters),
#    full 6-layer fits. Baseline first, then each structural term alone.
for cfg in "none:" "resid:--w_resid 1" "readout:--w_readout 1" \
           "readout_resid:--w_readout 1 --w_resid 1" "gramcross:--w_gram 1 --w_cross 1" \
           "all_new:--w_readout 1 --w_resid 1 --w_gram 1 --w_cross 1"; do
  name=${cfg%%:*}; flags=${cfg#*:}
  $P fit_mixers.py --run n_reg_$name --steps 4000 --groups 8 --d_hidden 512 \
     --eval_every 2000 $flags 2>&1 | tee logs/n_reg_$name.log
done

# 4. Best structural setting, long fit, to see if the deep-layer ceiling moves at all.
$P fit_mixers.py --run n_long_readout --steps 20000 --groups 8 --d_hidden 512 \
   --eval_every 5000 --w_readout 1 --w_resid 1 2>&1 | tee logs/n_long_readout.log

echo NIGHT_QUEUE_DONE
# 5. Give the card back to the chat model.
setsid nohup /home/bard/start_qwen38_iq4.sh > /tmp/qwen38_iq4.log 2>&1 < /dev/null &

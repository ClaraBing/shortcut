#!/bin/bash
# Launch training. Each setting below can be overridden by an environment variable, and any extra
# arguments are passed through to train.py as Hydra overrides. Evaluation runs during training.
#
# Usage:
#   bash scripts/train.sh
#   TASK=symmetric N_LAYER=6 LR=1e-4 bash scripts/train.sh
#   bash scripts/train.sh task.n=3 wandb.enabled=true

set -euo pipefail
cd "$(dirname "$0")/.."

# Task and data
TASK=${TASK:-parity}
LENGTH=${LENGTH:-32}            # training sequence length
EVAL_LENGTH=${EVAL_LENGTH:-null} # additional eval length; null = training length only

# Defaults for model size and optimization follow the parity (C2) setup in Liu et al. 2022,
# "Transformers Learn Shortcuts to Automata" (arXiv:2210.10749), Appendix B.

# Model size: 8 layers, 8 heads, embedding dim and MLP width 512
N_LAYER=${N_LAYER:-8}
N_HEAD=${N_HEAD:-8}
N_EMBD=${N_EMBD:-512}
MLP_RATIO=${MLP_RATIO:-4}

# Optimization: AdamW, lr in {3e-5, 1e-4, 3e-4}, weight decay 1e-4, batch size 16, EMA decay 0.9
LR=${LR:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-2}
BATCH_SIZE=${BATCH_SIZE:-64}
EMA_DECAY=${EMA_DECAY:-0.9}
MAX_STEPS=${MAX_STEPS:-10000}    # 600k fresh samples, the low end of the paper's 600k-5000k range
SEED=${SEED:-0}

LOG_EVERY=${LOG_EVERY:-1000}
EVAL_EVERY=${EVAL_EVERY:-1000}

python train.py \
  task=${TASK} \
  data.length=${LENGTH} \
  data.eval_length=${EVAL_LENGTH} \
  model.n_layer=${N_LAYER} \
  model.n_head=${N_HEAD} \
  model.n_embd=${N_EMBD} \
  model.mlp_ratio=${MLP_RATIO} \
  train.lr=${LR} \
  train.min_lr=$(python -c "print(${LR} / 10)") \
  train.weight_decay=${WEIGHT_DECAY} \
  train.batch_size=${BATCH_SIZE} \
  train.ema_decay=${EMA_DECAY} \
  train.max_steps=${MAX_STEPS} \
  train.log_every=${LOG_EVERY} \
  eval.every=${EVAL_EVERY} \
  seed=${SEED} \
  wandb.enabled=true \
  wandb.project=shortcut \
  wandb.name=${TASK}-${LENGTH}-${N_LAYER}-${N_HEAD}-${N_EMBD}-${LR}-${BATCH_SIZE}-${MAX_STEPS}-${SEED} \
  "$@"

# shortcut

Train GPT-style transformers to simulate automata, as in
[Transformers Learn Shortcuts to Automata](https://arxiv.org/abs/2210.10749).

Each task is a finite-state machine. The model reads a sequence of input symbols and predicts the state
(or a label of the state) at every position. Training data is generated fresh at every step.

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests
```

## Train

```bash
python train.py                                        # default: dihedral (D_8), length 100
python train.py task=symmetric task.n=5 data.length=64 model.n_layer=6
python train.py task=parity train.loss_last_only=true  # loss on the last position only
python train.py task=cyclic data.eval_length=200       # also evaluate at length 200
python train.py wandb.enabled=true wandb.project=my-project
```

`scripts/train.sh` wraps this with the main settings as environment variables. Extra arguments are
passed through as Hydra overrides:

```bash
TASK=symmetric N_LAYER=6 LR=1e-4 EVAL_LENGTH=200 bash scripts/train.sh wandb.enabled=true
```

Sweeps use Hydra multirun:

```bash
python train.py -m task=parity,cyclic,dihedral,quaternion model.n_layer=1,2,4,8 seed=0,1
```

wandb reads credentials from your environment: run `wandb login` once, or set `WANDB_API_KEY`.
Never put the key in configs or code.

See `configs/config.yaml` for all options and `configs/task/` for each task's parameters.
Outputs go to `outputs/<task>/<timestamp>/` (or `multirun/<timestamp>/<job>/`):

| File | Content |
|---|---|
| `metrics.jsonl` | train loss and eval metrics over time |
| `results.json` | metrics of the best and the last eval, incl. per-position accuracy |
| `best.pt`, `last.pt` | checkpoints (raw and, if enabled, EMA weights) |
| `.hydra/config.yaml` | the resolved config |

The best checkpoint is selected on `val/acc` (`val/acc_last` with `loss_last_only`).

## Evaluate

The training run always evaluates in-distribution (`val/*`, at `data.length`).
If `data.eval_length` is set, it also evaluates at that length (`val_len<L>/*`).
Reported metrics:

- `acc`: per-position accuracy.
- `acc_last`: accuracy at the last position.
- `acc_seq`: fraction of sequences that are correct at every position.
- `acc_per_pos`: accuracy at each position (in `results.json`).

To evaluate a saved checkpoint at another length:

```bash
python eval.py ckpt=outputs/dihedral/<run>/best.pt length=200 [use_ema=true]
```

Positions are learned embeddings, so the model only supports lengths up to `model.block_size`.
This defaults to `max(data.length, data.eval_length)`. Set it larger at training time if you plan to
evaluate on longer sequences later. Positions past the training length are never trained.

## Tasks

| `task=` | States | Inputs | Parameters (defaults) | `label_type` |
|---|---|---|---|---|
| `parity` | 2 | {0,1} | `prob1=0.5` | state |
| `cyclic` | Z_n | 0..n_actions-1 (add i) | `n=5, n_actions=2` | state |
| `dihedral` | D_2n (2n) | 0 = toggle, 1 = drive | `n=4` | state, toggle, position |
| `quaternion` | Q_8 (8) | {1,i,j,k} | | state |
| `symmetric` | S_n (n!) | identity, shift, swap, ... | `n=5, n_actions=3` | state, first_chair |
| `alternating` | A_n (n!/2) | identity, 3-cycles (1 2 x) | `n=5` | state, first_chair |
| `permutation_reset` | S_n (n!) | n! resets + generators | `n=5, generators=[12340, 10234]` | state |
| `flipflop` | n+1 | 0 = read, i = write i | `n=2, p_read=0.5` | state |
| `gridworld` | n | 0 = left, 1 = right | `n=9, prob1=0.5` | state, parity, boundary |
| `abab` | 5 | {0,1} | `prob_abab_pos_sample=0.25` | state, boundary |
| `add` | 2·n_addends | bits of all addends | `n_addends=2, prob1=0.5` | state, digit, carry |

## Code

- `shortcut/automata.py`: tasks. Each task is a transition table plus an input sampler, and sequences are
  generated in batches.
- `shortcut/model.py`: a nanoGPT-style transformer with a per-position classification head.
- `shortcut/utils.py`: model and task construction, loss, and evaluation.
- `train.py`, `eval.py`: Hydra entry points.

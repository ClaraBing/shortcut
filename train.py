"""Train a GPT-style model to simulate an automaton (per-position state prediction).

Usage:
    python train.py task=dihedral task.n=4 data.length=100 model.n_layer=4
    python train.py -m task=parity,cyclic,quaternion model.n_layer=1,2,4   # sweep
"""

import json
import math
import os
import time

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from shortcut.utils import (build_automaton, build_model, compute_loss, evaluate, get_device,
                            make_eval_set, set_seed, to_tensors)


def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))) * (cfg.lr - cfg.min_lr)


def run_evals(model, ema, eval_sets, cfg, device):
    models = {'': model}
    if ema is not None:
        models['ema/'] = ema.module
    results = {}
    for prefix, m in models.items():
        for split, eval_set in eval_sets.items():
            metrics = evaluate(m, eval_set, cfg.eval.batch_size, device, cfg.train.loss_last_only)
            results.update({f'{prefix}{split}/{k}': v for k, v in metrics.items()})
    return results


def scalars(results):
    return {k: v for k, v in results.items() if not isinstance(v, list)}


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    out_dir = HydraConfig.get().runtime.output_dir
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    automaton = build_automaton(cfg.task)
    model = build_model(cfg, automaton).to(device)
    print(f"Task {cfg.task.name}: vocab_size={automaton.vocab_size}, n_classes={automaton.n_classes}")
    print(f"Model: {model.num_params() / 1e6:.2f}M params, block_size={model.config.block_size}, device={device}")

    # Training data is sampled fresh at every step; eval sets are fixed and use separate seeds.
    train_rng = np.random.default_rng([cfg.seed, 0])
    eval_sets = {'val': make_eval_set(automaton, cfg.eval.n_samples, cfg.data.length, seed=[cfg.seed, 1])}
    eval_length = cfg.data.eval_length
    if eval_length and eval_length != cfg.data.length:
        eval_sets[f'val_len{eval_length}'] = make_eval_set(automaton, cfg.eval.n_samples, eval_length, seed=[cfg.seed, 2])
    select_key = 'val/acc_last' if cfg.train.loss_last_only else 'val/acc'

    tcfg = cfg.train
    optimizer = model.configure_optimizer(tcfg.weight_decay, tcfg.lr, tcfg.betas)
    ema = None
    if tcfg.ema_decay > 0:
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(tcfg.ema_decay), use_buffers=True)

    wandb_run = None
    if cfg.wandb.enabled:
        import wandb
        wandb_run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.wandb.name,
                               config=OmegaConf.to_container(cfg, resolve=True), dir=out_dir)

    metrics_file = open(os.path.join(out_dir, 'metrics.jsonl'), 'a')

    def log(record):
        metrics_file.write(json.dumps(record) + '\n')
        metrics_file.flush()
        if wandb_run is not None:
            wandb_run.log({k: v for k, v in record.items() if k != 'step'}, step=record['step'])

    def save_checkpoint(name, step, results):
        torch.save({
            'model': model.state_dict(),
            'ema': ema.module.state_dict() if ema is not None else None,
            'model_config': vars(model.config),
            'cfg': OmegaConf.to_container(cfg, resolve=True),
            'step': step,
            'results': results,
        }, os.path.join(out_dir, name))

    best = {'step': 0, 'score': -1.0, 'results': None}
    evals_since_best = 0
    t0 = time.time()
    model.train()
    for step in range(1, tcfg.max_steps + 1):
        lr = get_lr(step - 1, tcfg)
        for group in optimizer.param_groups:
            group['lr'] = lr

        x, y = to_tensors(*automaton.sample(train_rng, tcfg.batch_size, cfg.data.length), device)
        loss = compute_loss(model(x), y, tcfg.loss_last_only)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update_parameters(model)

        if step % tcfg.log_every == 0:
            log({'step': step, 'train/loss': loss.item(), 'train/lr': lr})

        if step % cfg.eval.every == 0 or step == tcfg.max_steps:
            results = run_evals(model, ema, eval_sets, cfg, device)
            log({'step': step, **scalars(results)})
            print(f"step {step:6d} | loss {loss.item():.4f} | " + ' | '.join(
                f"{k} {v:.4f}" for k, v in scalars(results).items() if k.endswith(('/acc', '/acc_seq'))
            ) + f" | {time.time() - t0:.0f}s", flush=True)

            if results[select_key] > best['score']:
                best = {'step': step, 'score': results[select_key], 'results': results}
                evals_since_best = 0
                save_checkpoint('best.pt', step, results)
            else:
                evals_since_best += 1
            save_checkpoint('last.pt', step, results)

            if tcfg.patience is not None and evals_since_best >= tcfg.patience:
                print(f"Early stopping at step {step}: no improvement in {select_key} for {tcfg.patience} evals.")
                break

    summary = {'best_step': best['step'], 'best': best['results'], 'last_step': step, 'last': results}
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Best {select_key} = {best['score']:.4f} at step {best['step']}. Results saved to {out_dir}")

    metrics_file.close()
    if wandb_run is not None:
        wandb_run.summary.update({f'best/{k}': v for k, v in scalars(best['results']).items()})
        wandb_run.finish()


if __name__ == '__main__':
    main()

"""Evaluate a trained checkpoint, optionally at a different sequence length.

Usage:
    python eval.py ckpt=outputs/dihedral/<run>/best.pt length=200
"""

import json
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from shortcut.model import GPT, GPTConfig
from shortcut.utils import build_automaton, evaluate, get_device, make_eval_set


@hydra.main(version_base='1.3', config_path='configs', config_name='eval')
def main(cfg: DictConfig):
    device = get_device(cfg.device)
    ckpt = torch.load(cfg.ckpt, map_location=device, weights_only=False)
    train_cfg = OmegaConf.create(ckpt['cfg'])

    model = GPT(GPTConfig(**ckpt['model_config'])).to(device)
    if cfg.use_ema:
        assert ckpt['ema'] is not None, "This checkpoint was trained without EMA."
        model.load_state_dict(ckpt['ema'])
    else:
        model.load_state_dict(ckpt['model'])

    automaton = build_automaton(train_cfg.task)
    length = cfg.length or train_cfg.data.length
    eval_set = make_eval_set(automaton, cfg.n_samples, length, seed=cfg.seed)
    results = evaluate(model, eval_set, cfg.batch_size, device, train_cfg.train.loss_last_only)

    print(f"{train_cfg.task.name} (trained at length {train_cfg.data.length}, step {ckpt['step']}), "
          f"eval length {length}: " + ', '.join(f"{k}={v:.4f}" for k, v in results.items() if k != 'acc_per_pos'))
    out_path = os.path.join(os.path.dirname(os.path.abspath(cfg.ckpt)),
                            f"eval_{os.path.basename(cfg.ckpt).removesuffix('.pt')}{'_ema' if cfg.use_ema else ''}_len{length}.json")
    with open(out_path, 'w') as f:
        json.dump({'ckpt': cfg.ckpt, 'length': length, 'use_ema': cfg.use_ema, **results}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()

import random

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from shortcut.automata import get_automaton
from shortcut.model import GPT, GPTConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device):
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def build_automaton(task_cfg):
    kwargs = OmegaConf.to_container(task_cfg, resolve=True)
    return get_automaton(kwargs.pop('name'), **kwargs)


def build_model(cfg, automaton):
    block_size = cfg.model.block_size or max(cfg.data.length, cfg.data.eval_length or 0)
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_kwargs.pop('block_size')
    config = GPTConfig(vocab_size=automaton.vocab_size, n_classes=automaton.n_classes,
                       block_size=block_size, **model_kwargs)
    return GPT(config)


def to_tensors(x, y, device):
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def compute_loss(logits, y, loss_last_only=False):
    if loss_last_only:
        logits, y = logits[:, -1], y[:, -1]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def make_eval_set(automaton, n_samples, length, seed):
    """A fixed eval set, generated once from its own seed."""
    return automaton.sample(np.random.default_rng(seed), n_samples, length)


@torch.no_grad()
def evaluate(model, eval_set, batch_size, device, loss_last_only=False):
    """Returns the loss and accuracies on a fixed eval set.

    acc: per-position accuracy over all positions.
    acc_last: accuracy at the last position.
    acc_seq: fraction of sequences with every position correct.
    acc_per_pos: per-position accuracy (list of length T).
    """
    was_training = model.training
    model.eval()
    x_all, y_all = eval_set
    n = len(x_all)
    loss_sum, correct = 0.0, None
    for i in range(0, n, batch_size):
        x, y = to_tensors(x_all[i:i + batch_size], y_all[i:i + batch_size], device)
        logits = model(x)
        loss_sum += compute_loss(logits, y, loss_last_only).item() * len(x)
        batch_correct = (logits.argmax(-1) == y).cpu()
        correct = batch_correct if correct is None else torch.cat([correct, batch_correct])
    model.train(was_training)
    correct = correct.float()
    return {
        'loss': loss_sum / n,
        'acc': correct.mean().item(),
        'acc_last': correct[:, -1].mean().item(),
        'acc_seq': correct.all(1).float().mean().item(),
        'acc_per_pos': correct.mean(0).tolist(),
    }

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TINY = ['model.n_layer=1', 'model.n_embd=32', 'model.n_head=2', 'train.max_steps=10',
        'eval.every=5', 'eval.n_samples=32', 'data.length=8']


def run(script, *overrides):
    # MKL_THREADING_LAYER may be set by numpy's MKL in this process, which breaks torch in the subprocess.
    env = {k: v for k, v in os.environ.items() if k != 'MKL_THREADING_LAYER'}
    proc = subprocess.run([sys.executable, script, *overrides], cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize('extra', [[], ['train.loss_last_only=true', 'train.ema_decay=0.9'], ['model.pos_emb=learned']])
def test_train_and_eval(tmp_path, extra):
    run('train.py', 'task=dihedral', 'data.eval_length=12', f'hydra.run.dir={tmp_path}', *TINY, *extra)
    results = json.loads((tmp_path / 'results.json').read_text())
    assert 'val_len12/acc' in results['last']
    assert len(results['last']['val_len12/acc_per_pos']) == 12
    run('eval.py', f'ckpt={tmp_path / "best.pt"}', 'length=12')
    assert (tmp_path / 'eval_best_len12.json').exists()

"""Checks the transition-table automata against closed-form solutions.

Equivalence with the HF reference (https://huggingface.co/datasets/synthseq/automata) was verified
for all tasks when porting; these tests guard the main semantics without depending on that file.
"""

import numpy as np
import pytest

from shortcut.automata import TASKS, get_automaton

B, T = 32, 40


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize('name', sorted(TASKS))
def test_shapes_and_ranges(name, rng):
    automaton = get_automaton(name)
    x, y = automaton.sample(rng, B, T)
    assert x.shape == y.shape == (B, T)
    assert x.dtype == y.dtype == np.int64
    assert 0 <= x.min() and x.max() < automaton.vocab_size
    assert 0 <= y.min() and y.max() < automaton.n_classes


def test_parity(rng):
    x, y = get_automaton('parity').sample(rng, B, T)
    np.testing.assert_array_equal(y, np.cumsum(x, 1) % 2)


def test_cyclic(rng):
    x, y = get_automaton('cyclic', n=7, n_actions=4).sample(rng, B, T)
    np.testing.assert_array_equal(y, np.cumsum(x, 1) % 7)


def test_dihedral(rng):
    n = 5
    automaton = get_automaton('dihedral', n=n)
    x, y = automaton.sample(rng, B, T)
    toggle = np.cumsum(x == 0, 1) % 2
    position = np.cumsum((x != 0) * (-1) ** toggle, 1) % n
    np.testing.assert_array_equal(y, n * toggle + position)


def test_gridworld(rng):
    n = 4
    x, y = get_automaton('gridworld', n=n).sample(rng, B, T)
    for xb, yb in zip(x, y):
        s, expected = 0, []
        for a in xb:
            s = min(max(s + (1 if a else -1), 0), n - 1)
            expected.append(s)
        np.testing.assert_array_equal(yb, expected)


def test_add(rng):
    automaton = get_automaton('add', n_addends=2, label_type='digit')
    x, y = automaton.sample(rng, B, T)
    weights = 2 ** np.arange(T, dtype=object)
    for xb, yb in zip(x, y):
        a, b = (xb & 1).astype(object), ((xb >> 1) & 1).astype(object)
        assert (a * weights).sum() + (b * weights).sum() == (yb.astype(object) * weights).sum()


def test_flipflop(rng):
    x, y = get_automaton('flipflop', n=3).sample(rng, B, T)
    for xb, yb in zip(x, y):
        s, expected = 0, []
        for a in xb:
            s = a if a else s
            expected.append(s)
        np.testing.assert_array_equal(yb, expected)


@pytest.mark.parametrize('name,kwargs,order', [
    ('symmetric', dict(n=4), 24), ('alternating', dict(n=5), 60), ('quaternion', {}, 8),
    ('dihedral', dict(n=4), 8), ('permutation_reset', dict(n=4, generators=['1230', '1023']), 24),
])
def test_group_states(name, kwargs, order, rng):
    automaton = get_automaton(name, **kwargs)
    assert automaton.n_states == order
    # every reachable state is visited and the transition table is a bijection per (non-reset) action
    _, y = automaton.sample(rng, 256, 200)
    assert len(np.unique(y)) == order
    for a in range(automaton.n_actions):
        column = automaton.transition[:, a]
        assert len(np.unique(column)) in (1, order)

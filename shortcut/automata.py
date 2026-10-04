"""Automata tasks, generated on the fly.

Ported from https://huggingface.co/datasets/synthseq/automata (automata.py).
Every task is a deterministic finite-state machine, so each one is represented by
  - a transition table `transition[state, action] -> next_state`,
  - an initial state,
  - an input sampler, and
  - label maps `label_maps[label_type][state] -> label`.
Labels are given after reading each input token, so `y[:, t]` is a function of `x[:, :t+1]`.

Differences from the HF reference:
- add: sequences have length `length` and the last input is the zero pad, so the final carry is
  written out. HF appends the pad, so its sequences are one token longer.
- symmetric with n_actions > 3: the swap generator is kept. HF overwrites it.
- gridworld: the 'parity' and 'boundary' label types are implemented. HF documents them but
  always returns the state.
- flipflop and dihedral with label_type='position' work. HF errors or returns None for these.
- Variable-length sampling (random_length) is not supported.
"""

import itertools
import math

import numpy as np


class Automaton:
    def __init__(self, transition, init_state=0, label_type='state', label_maps=None):
        self.transition = np.asarray(transition, dtype=np.int64)
        self.n_states, self.n_actions = self.transition.shape
        self.init_state = init_state
        label_maps = dict(label_maps or {})
        label_maps.setdefault('state', np.arange(self.n_states))
        if label_type not in label_maps:
            raise ValueError(f"label_type={label_type!r} not in {sorted(label_maps)}")
        self.label_type = label_type
        self.label_map = np.asarray(label_maps[label_type], dtype=np.int64)
        self.n_classes = int(self.label_map.max()) + 1

    @property
    def vocab_size(self):
        return self.n_actions

    def sample_inputs(self, rng, batch_size, length):
        """Default: i.i.d. uniform actions."""
        return rng.integers(self.n_actions, size=(batch_size, length))

    def run(self, x):
        """Returns the state after each step, shape (batch, length)."""
        states = np.empty_like(x)
        s = np.full(x.shape[0], self.init_state, dtype=np.int64)
        for t in range(x.shape[1]):
            s = self.transition[s, x[:, t]]
            states[:, t] = s
        return states

    def sample(self, rng, batch_size, length):
        x = self.sample_inputs(rng, batch_size, length)
        return x, self.label_map[self.run(x)]


def _binary_inputs(rng, batch_size, length, prob1):
    return (rng.random((batch_size, length)) < prob1).astype(np.int64)


class Parity(Automaton):
    def __init__(self, prob1=0.5, label_type='state'):
        super().__init__([[0, 1], [1, 0]], label_type=label_type)
        self.prob1 = prob1

    def sample_inputs(self, rng, batch_size, length):
        return _binary_inputs(rng, batch_size, length, self.prob1)


class Gridworld(Automaton):
    """1D gridworld with states 0..n-1; action 0 moves left, 1 moves right (clamped at the boundaries)."""
    def __init__(self, n=9, prob1=0.5, label_type='state'):
        S = n - 1
        transition = [[max(s - 1, 0), min(s + 1, S)] for s in range(n)]
        states = np.arange(n)
        label_maps = {
            'parity': states % 2,
            'boundary': ((states == 0) | (states == S)).astype(np.int64),
        }
        super().__init__(transition, label_type=label_type, label_maps=label_maps)
        self.prob1 = prob1

    def sample_inputs(self, rng, batch_size, length):
        return _binary_inputs(rng, batch_size, length, self.prob1)


class ABAB(Automaton):
    """Recognizes (01)*: 4 states + 1 absorbing (failure) state, starting from state 3."""
    def __init__(self, prob1=0.5, prob_abab_pos_sample=0.25, label_type='state'):
        transition = [
            [4, 1],  # state 0
            [2, 4],  # state 1
            [4, 3],  # state 2
            [0, 4],  # state 3
            [4, 4],  # state 4
        ]
        label_maps = {'boundary': (np.arange(5) == 3).astype(np.int64)}
        super().__init__(transition, init_state=3, label_type=label_type, label_maps=label_maps)
        self.prob1 = prob1
        self.prob_abab_pos_sample = prob_abab_pos_sample

    def sample_inputs(self, rng, batch_size, length):
        x = _binary_inputs(rng, batch_size, length, self.prob1)
        pos = rng.random(batch_size) < self.prob_abab_pos_sample
        x[pos] = np.arange(length) % 2
        return x


class Adder(Automaton):
    """Adds `n_addends` binary numbers, least significant bit first.

    The input token at each position encodes the bits of all addends (bit i = addend i).
    The state is (digit, carry), with id `digit + n_addends * carry`.
    The last position is always 0, so that the final carry is written out.
    """
    def __init__(self, n_addends=2, prob1=0.5, label_type='state'):
        k = n_addends
        transition = np.zeros((2 * k, 2 ** k), dtype=np.int64)
        for state in range(2 * k):
            carry = state // k
            for token in range(2 ** k):
                total = bin(token).count('1') + carry
                transition[state, token] = total % k + k * (total // k)
        states = np.arange(2 * k)
        label_maps = {'digit': states % k, 'carry': states // k}
        super().__init__(transition, label_type=label_type, label_maps=label_maps)
        self.n_addends = n_addends
        self.prob1 = prob1

    def sample_inputs(self, rng, batch_size, length):
        bits = rng.random((batch_size, length, self.n_addends)) < self.prob1
        bits[:, -1] = 0
        return (bits * (2 ** np.arange(self.n_addends))).sum(-1).astype(np.int64)


class FlipFlop(Automaton):
    """Action 0 reads (keeps the state); action i in 1..n writes i. The start state is 0."""
    def __init__(self, n=2, p_read=0.5, label_type='state'):
        transition = [[s] + list(range(1, n + 1)) for s in range(n + 1)]
        super().__init__(transition, label_type=label_type)
        self.n = n
        self.p_read = p_read

    def sample_inputs(self, rng, batch_size, length):
        writes = rng.integers(1, self.n + 1, size=(batch_size, length))
        return writes * (rng.random((batch_size, length)) >= self.p_read)


def _permutation_table(perms, actions):
    """perms: list of state permutations (tuples). actions: list of index arrays; state -> state[idx]."""
    perm2id = {p: i for i, p in enumerate(perms)}
    return [[perm2id[tuple(np.asarray(p)[list(a)])] for a in actions] for p in perms]


def _is_even(perm):
    n_inversions = sum(perm[i] > perm[j] for i in range(len(perm)) for j in range(i + 1, len(perm)))
    return n_inversions % 2 == 0


class Symmetric(Automaton):
    """S_n. Actions: 0 = identity, 1 = shift by 1, 2 = swap the first two; further actions are
    the remaining permutations in itertools order."""
    def __init__(self, n=5, n_actions=3, label_type='state'):
        perms = list(itertools.permutations(range(n)))
        identity = tuple(range(n))
        shift = tuple(range(1, n)) + (0,)
        swap = (1, 0) + tuple(range(2, n))
        actions = [identity, shift, swap]
        actions += [p for p in perms if p not in actions]
        actions = actions[:n_actions]
        label_maps = {'first_chair': [p[0] for p in perms]}
        super().__init__(_permutation_table(perms, actions), label_type=label_type, label_maps=label_maps)


class Alternating(Automaton):
    """A_n. Actions: 0 = identity, and the 3-cycles (1 2 x) for x = 3..n."""
    def __init__(self, n=5, label_type='state'):
        perms = [p for p in itertools.permutations(range(n)) if _is_even(p)]
        actions = [list(range(n))]
        for i in range(2, n):
            idx = list(range(n))
            idx[0], idx[1], idx[i] = 1, i, 0
            actions.append(idx)
        label_maps = {'first_chair': [p[0] for p in perms]}
        super().__init__(_permutation_table(perms, actions), label_type=label_type, label_maps=label_maps)


class Cyclic(Automaton):
    """Z_n. Action i adds i (mod n), for i in 0..n_actions-1."""
    def __init__(self, n=5, n_actions=2, label_type='state'):
        transition = [[(s + a) % n for a in range(n_actions)] for s in range(n)]
        super().__init__(transition, label_type=label_type)


class Dihedral(Automaton):
    """D_2n. Action 0 toggles the direction; action 1 moves one step on the n-cycle in the current direction.
    The state id is `n * toggle + position`."""
    def __init__(self, n=4, label_type='state'):
        transition = []
        for state in range(2 * n):
            toggle, pos = divmod(state, n)
            step = 1 if toggle == 0 else -1
            transition.append([n * (1 - toggle) + pos, n * toggle + (pos + step) % n])
        states = np.arange(2 * n)
        label_maps = {'toggle': states // n, 'position': states % n}
        super().__init__(transition, label_type=label_type, label_maps=label_maps)


class Quaternion(Automaton):
    """Q_8. States: {1, i, j, k} x {+, -} (ids 0-3 positive, 4-7 negative); actions: {1, i, j, k}."""
    def __init__(self, label_type='state'):
        pos = [
            [0, 1, 2, 3],
            [1, 4, 3, 6],
            [2, 7, 4, 1],
            [3, 2, 5, 4],
        ]
        neg = [[(s + 4) % 8 for s in row] for row in pos]
        super().__init__(pos + neg, label_type=label_type)


class PermutationReset(Automaton):
    """S_n with resets. Actions 0..n!-1 reset the state to that permutation;
    action n!+g applies generator g (state -> generator[state]).
    Resets occur at the first position and then at gaps drawn uniformly from {1, 2, 4, ...} (< length)."""
    def __init__(self, n=5, generators=('12340', '10234'), perm_probs=None, label_type='state'):
        generators = [np.array([int(c) for c in g]) if isinstance(g, str) else np.asarray(g) for g in generators]
        perms = list(itertools.permutations(range(n)))
        perm2id = {p: i for i, p in enumerate(perms)}
        n_perms = len(perms)
        assert n_perms == math.factorial(n)
        transition = [list(range(n_perms)) + [perm2id[tuple(g[list(p)])] for g in generators] for p in perms]
        super().__init__(transition, label_type=label_type)
        self.n_perms = n_perms
        self.n_generators = len(generators)
        self.perm_probs = perm_probs

    def sample_inputs(self, rng, batch_size, length):
        x = rng.choice(self.n_generators, p=self.perm_probs, size=(batch_size, length)) + self.n_perms
        lags = [1]
        while lags[-1] * 2 < length:
            lags.append(lags[-1] * 2)
        gaps = rng.choice(lags, size=(batch_size, length))
        reset_pos = np.concatenate([np.zeros((batch_size, 1), dtype=np.int64), gaps[:, :-1].cumsum(1)], 1)
        rows = np.broadcast_to(np.arange(batch_size)[:, None], reset_pos.shape)
        mask = reset_pos < length
        x[rows[mask], reset_pos[mask]] = rng.integers(self.n_perms, size=mask.sum())
        return x


TASKS = {
    'abab': ABAB,
    'add': Adder,
    'alternating': Alternating,
    'cyclic': Cyclic,
    'dihedral': Dihedral,
    'flipflop': FlipFlop,
    'gridworld': Gridworld,
    'parity': Parity,
    'quaternion': Quaternion,
    'symmetric': Symmetric,
    'permutation_reset': PermutationReset,
}


def get_automaton(name, **kwargs):
    if name not in TASKS:
        raise ValueError(f"Unknown task {name!r}; choose from {sorted(TASKS)}")
    return TASKS[name](**kwargs)

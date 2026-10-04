"""GPT-2-style transformer, adapted from nanoGPT (https://github.com/karpathy/nanoGPT).

The language-modeling head is replaced by a per-position classification head
(no weight tying, since inputs and labels have different vocabularies).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    n_classes: int
    block_size: int
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 512
    mlp_ratio: int = 4
    dropout: float = 0.0
    bias: bool = True
    causal: bool = True
    pos_emb: str = 'learned'  # learned | sinusoidal; checkpoints saved before this option existed are 'learned'


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


def sinusoidal_pos_emb(T, d, device):
    """Fixed sinusoidal position encodings (Vaswani et al., 2017): (T, d)."""
    pos = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
    inv_freq = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * inv_freq)
    pe[:, 1::2] = torch.cos(pos * inv_freq[:d // 2])
    return pe


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.causal = config.causal

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.pos_emb in ('learned', 'sinusoidal'), f"Unknown pos_emb: {config.pos_emb}"
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            **({'wpe': nn.Embedding(config.block_size, config.n_embd)} if config.pos_emb == 'learned' else {}),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.head = nn.Linear(config.n_embd, config.n_classes, bias=False)

        self.apply(self._init_weights)
        # scaled init for the residual projections, per GPT-2
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx):
        """idx: (B, T) input tokens -> logits: (B, T, n_classes)."""
        T = idx.size(1)
        if self.config.pos_emb == 'learned':
            assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"
            pos_emb = self.transformer.wpe(torch.arange(T, dtype=torch.long, device=idx.device))
        else:  # computed on the fly, so any length works (e.g. length generalization at eval time)
            pos_emb = sinusoidal_pos_emb(T, self.config.n_embd, idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        return self.head(self.transformer.ln_f(x))

    def configure_optimizer(self, weight_decay, learning_rate, betas):
        """AdamW; weight decay on 2D params (matmuls, embeddings) only."""
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {'params': [p for p in params if p.dim() >= 2], 'weight_decay': weight_decay},
            {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=tuple(betas))

"""
HSTUBlock and HardConcreteGate — built matching model.py (the real GDELT
reference) exactly, not the "canonical" 3-block/2-gate description from the
working doc. Per explicit instruction: reconcile against model.py as-is.

IMPORTANT ARCHITECTURAL NOTE, worth reading before building prediction heads:
model.py's GDELTEncoder uses causal=False for BOTH HSTU blocks, and the
model's output head is MLMHead — a masked-language-model-style head
predicting a vocab distribution at every position, not specifically a
next-token/next-event head. That means the actual GDELT training paradigm
is masked prediction (bidirectional context, some positions randomly
masked and predicted), not causal next-event forecasting.

This matters for Companies House: the working doc's §1/§7 describe
"next filing event" prediction, which reads as autoregressive/causal. If
we're reconciling with model.py's actual approach, the CH model should
likely also be MLM-style (non-causal, mask-and-predict) rather than
causal next-event prediction — but this is a real training-paradigm
decision, not just a structural detail, and hasn't been decided yet.
Flagging here rather than silently picking one when building prediction
heads.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSTUBlock(nn.Module):
    """
    Matches model.py's HSTUBlock exactly: pre-norm attention with optional
    causal masking (default False, matching GDELTEncoder's actual usage),
    SwiGLU FFN (gate * up -> down), residual connections around both.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_gate = nn.Linear(d_model, d_model * 4)
        self.ffn_up = nn.Linear(d_model, d_model * 4)
        self.ffn_down = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                                   # [B, L, D]
        key_padding_mask: Optional[torch.Tensor] = None,    # [B, L] bool, True = real token
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_mask = None
        if self.causal:
            L = x.size(1)
            attn_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        x, _ = self.attn(
            x, x, x,
            key_padding_mask=~key_padding_mask if key_padding_mask is not None else None,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = self.dropout(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))
        return self.dropout(x) + residual


class HardConcreteGate(nn.Module):
    """
    Matches model.py's HardConcreteGate exactly: one log_alpha parameter
    per sequence position (not input-conditioned — that's flagged in the
    working doc as a future step, not yet built here or in model.py).

    Stochastic (binary-concrete relaxation) during training, deterministic
    sigmoid threshold at eval. l0_loss is the standard L0 sparsity
    penalty used to push the gate toward the target sparsity.
    """

    def __init__(self, seq_len: int, beta: float = 2.0 / 3.0, zeta: float = -0.1, gamma: float = 1.1):
        super().__init__()
        self.beta = beta
        self.zeta = zeta
        self.gamma = gamma
        self.log_alpha = nn.Parameter(torch.full((seq_len,), -3.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        log_alpha = self.log_alpha[:L]

        if self.training:
            u = torch.zeros_like(log_alpha).uniform_().clamp(1e-8, 1 - 1e-8)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / self.beta)
        else:
            s = torch.sigmoid(log_alpha / self.beta)

        gates = (s * (self.gamma - self.zeta) + self.zeta).clamp(0.0, 1.0)
        return x * gates.unsqueeze(0).unsqueeze(-1).expand(B, -1, D), gates.unsqueeze(0).expand(B, -1)

    def l0_loss(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha - self.beta * math.log(-self.zeta / self.gamma)).sum()

"""
HSTUBlock and HardConcreteGate, per CORTEX_ARCHITECTURE.md (the canonical
spec, confirmed against model_v2.py — the actual current GDELT
implementation). Supersedes the earlier reconciliation against model.py,
which turned out to be the stale v1 file: this uses 3 HSTU blocks + 2
Hard Concrete gates (not 2 blocks + 1 gate), and DecayedSinusoidalPE is
confirmed correct (see model/embeddings.py), not model.py's plain learned
positional embedding.

Training paradigm note (still applies): causal MLM decided for Companies
House (mask random events, predict them, causal attention throughout —
consistent with §9's no-future-leakage principle). The canonical doc's own
causal mask requirement ("Position i never attends to j > i. No future
leakage.") is fully compatible with this.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSTUBlock(nn.Module):
    """
    Pre-norm causal self-attention + SwiGLU FFN, per §2.2. Causal masking
    is a hard requirement per the canonical spec (not optional/toggleable
    the way model.py had it) — this keeps the causal flag for testing
    flexibility, but production use should always pass causal=True.

    Optionally returns attention weights (mean across heads) for §2.5's
    attribution mechanism — set return_attn=True. Off by default since
    training doesn't need it and computing/returning weights has overhead.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_gate = nn.Linear(d_model, d_model * 4)
        self.ffn_up = nn.Linear(d_model, d_model * 4)
        self.ffn_down = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self._mask_cache: Optional[torch.Tensor] = None

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        if self._mask_cache is None or self._mask_cache.size(0) != L or self._mask_cache.device != device:
            self._mask_cache = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        return self._mask_cache

    def forward(
        self,
        x: torch.Tensor,                                   # [B, L, D]
        key_padding_mask: Optional[torch.Tensor] = None,    # [B, L] bool, True = real token
        return_attn: bool = False,
    ):
        residual = x
        x = self.norm1(x)
        attn_mask = self._causal_mask(x.size(1), x.device) if self.causal else None
        x, attn_weights = self.attn(
            x, x, x,
            key_padding_mask=~key_padding_mask if key_padding_mask is not None else None,
            attn_mask=attn_mask,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        # Defense-in-depth: combining a causal attn_mask with key_padding_mask
        # can produce an all-masked attention row for certain (batch, query)
        # pairs in some PyTorch versions, which softmaxes to NaN (0/0 from an
        # all -inf row). Zero out any NaN before it propagates into the
        # residual stream and corrupts the rest of the forward/backward pass.
        x = torch.nan_to_num(x, nan=0.0)
        x = self.dropout(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))
        x = self.dropout(x) + residual

        if return_attn:
            return x, attn_weights
        return x, None


class HardConcreteGate(nn.Module):
    """
    Per CORTEX_ARCHITECTURE.md §2.3. One log_alpha parameter per sequence
    position (fixed, not input-conditioned — that's §8's documented future
    step, not built yet, here or in the canonical reference). Stochastic
    (binary-concrete relaxation) during training, deterministic sigmoid
    threshold at eval.

    init_log_alpha matters: -3.0 gives a near-closed gate at init (used for
    Gate 1's normal training start). Gate 2 in Phase 1 is frozen at exactly
    0.0 (~0.5 gate value, pass-through) rather than closed — see
    freeze_gate_at_zero / two-phase training in train.py.
    """

    def __init__(
        self, seq_len: int, beta: float = 2.0 / 3.0, zeta: float = -0.1, gamma: float = 1.1,
        init_log_alpha: float = -3.0,
    ):
        super().__init__()
        self.beta = beta
        self.zeta = zeta
        self.gamma = gamma
        self.log_alpha = nn.Parameter(torch.full((seq_len,), init_log_alpha))

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
        """Sum over positions — a rate, not yet normalized (caller divides by L)."""
        return torch.sigmoid(self.log_alpha - self.beta * math.log(-self.zeta / self.gamma)).sum()

    def freeze_at_zero(self) -> None:
        """Phase 1: Gate 2 frozen at log_alpha=0.0 (~0.5 gate value, pass-through, no selection)."""
        with torch.no_grad():
            self.log_alpha.fill_(0.0)
        self.log_alpha.requires_grad = False

    def binarize_and_freeze(self, threshold: float = 0.30) -> None:
        """
        Phase 1 -> Phase 2 transition for Gate 1: evaluate deterministically,
        then hard-set log_alpha to +5.0 (open) or -5.0 (closed) per position
        based on which side of `threshold` (0.30, confirmed against
        CORTEX_ARCHITECTURE.md's freeze_at_inference_state) the eval-mode
        gate value fell on, and freeze. Gate 1 becomes a fixed, binary
        selection from this point.
        """
        with torch.no_grad():
            was_training = self.training
            self.eval()
            s = torch.sigmoid(self.log_alpha / self.beta)
            gate_vals = (s * (self.gamma - self.zeta) + self.zeta).clamp(0.0, 1.0)
            self.log_alpha.copy_(torch.where(gate_vals > threshold, 5.0, -5.0))
            if was_training:
                self.train()
        self.log_alpha.requires_grad = False

    def unfreeze(self) -> None:
        self.log_alpha.requires_grad = True

    def deterministic_gate_values(self, seq_len: int) -> torch.Tensor:
        """
        Eval-style (deterministic) gate values regardless of self.training —
        used for dynamic_lambda's frac_above_threshold, which per the
        canonical spec is computed "at inference" even during a training
        step. Decoupled from forward()'s stochastic training path.
        """
        with torch.no_grad():
            log_alpha = self.log_alpha[:seq_len]
            s = torch.sigmoid(log_alpha / self.beta)
            return (s * (self.gamma - self.zeta) + self.zeta).clamp(0.0, 1.0)


def dynamic_lambda(gate_values: torch.Tensor, threshold: float = 0.70) -> torch.Tensor:
    """
    Per CORTEX_ARCHITECTURE.md §2.3: lambda_k = 2*(sigmoid(4*frac_above_threshold) - 0.5),
    computed fresh each batch from the gate's own eval-mode activations.
    frac_above_threshold = fraction of this gate's positions with value > threshold.

    Bidirectional S-curve: 0% open -> lambda=0 (no sparsity pressure, pure
    MLM); 100% open -> lambda~0.96 (strong pressure to close). Self-scaling:
    as the gate approaches its target, pressure naturally moderates.
    """
    frac_above = (gate_values > threshold).float().mean()
    return 2.0 * (torch.sigmoid(4.0 * frac_above) - 0.5)

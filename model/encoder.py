"""
Companies House encoder — 3 HSTU blocks + 2 Hard Concrete gates, per
CORTEX_ARCHITECTURE.md §2.2:

    Input embeddings
      -> HSTU Block 1 (causal self-attention + SwiGLU FFN)
      -> Hard Concrete Gate 1 (coarse selection, target ~15%)
      -> HSTU Block 2 (causal self-attention + SwiGLU FFN)
      -> Hard Concrete Gate 2 (fine selection, target ~5%)
      -> HSTU Block 3 (causal self-attention + SwiGLU FFN)
      -> LayerNorm

Supersedes the earlier 2-block/1-gate version built against model.py
(confirmed stale — see hstu.py's module docstring).
"""
from typing import Optional

import torch
import torch.nn as nn

from model.config import EmbeddingConfig
from model.embeddings import InputEmbeddingStack
from model.hstu import HSTUBlock, HardConcreteGate


class CompaniesHouseEncoder(nn.Module):
    def __init__(
        self,
        n_categories: int,
        n_types: int,
        n_subtypes: int,
        n_companies: int,
        cfg: EmbeddingConfig,
        n_heads: int = 4,
        max_seq_len: int = 512,
        gate1_target: float = 0.15,
        gate2_target: float = 0.05,
        gate_beta: float = 2.0 / 3.0,
        gate_zeta: float = -0.1,
        gate_gamma: float = 1.1,
    ):
        super().__init__()
        self.gate1_target = gate1_target
        self.gate2_target = gate2_target

        self.input_stack = InputEmbeddingStack(n_categories, n_types, n_subtypes, n_companies, cfg)
        self.hstu1 = HSTUBlock(cfg.embed_dim, n_heads, cfg.dropout, causal=True)
        self.gate1 = HardConcreteGate(max_seq_len, gate_beta, gate_zeta, gate_gamma, init_log_alpha=-3.0)
        self.hstu2 = HSTUBlock(cfg.embed_dim, n_heads, cfg.dropout, causal=True)
        self.gate2 = HardConcreteGate(max_seq_len, gate_beta, gate_zeta, gate_gamma, init_log_alpha=-3.0)
        self.hstu3 = HSTUBlock(cfg.embed_dim, n_heads, cfg.dropout, causal=True)
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(
        self,
        category_ids: torch.Tensor,
        type_ids: torch.Tensor,
        subtype_ids: torch.Tensor,
        company_ids: torch.Tensor,
        position_or_elapsed: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> dict:
        x = self.input_stack(
            category_ids, type_ids, subtype_ids, company_ids, position_or_elapsed, attention_mask
        )
        x, attn1 = self.hstu1(x, key_padding_mask=attention_mask, return_attn=return_attn)
        x, gates1 = self.gate1(x)
        x, attn2 = self.hstu2(x, key_padding_mask=attention_mask, return_attn=return_attn)
        x, gates2 = self.gate2(x)
        x, attn3 = self.hstu3(x, key_padding_mask=attention_mask, return_attn=return_attn)

        return {
            "hidden": self.norm(x),
            "gates1": gates1,
            "gates2": gates2,
            "attn1": attn1,
            "attn2": attn2,
            "attn3": attn3,
        }

    def l0_rate(self, gate: HardConcreteGate, seq_len: int) -> torch.Tensor:
        """Normalized L0 (open-fraction), not the raw positional sum."""
        gate_len = min(seq_len, gate.log_alpha.shape[0])
        return gate.l0_loss() / gate_len

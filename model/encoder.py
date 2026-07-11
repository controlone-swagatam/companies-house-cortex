"""
Companies House encoder. Matches GDELTEncoder's structure exactly:
embed -> hstu1 -> gate -> hstu2 -> final norm. 2 HSTU blocks, 1 gate
(not the documented 3-block/2-gate design — see hstu.py's module docstring
for why this is deliberate, not an oversight).

Positional/entity embedding is handled inside InputEmbeddingStack already
(model/embeddings.py), so this encoder's forward is simpler than
GDELTEncoder's — the +pos_embed / +country_embed steps are internal to the
stack rather than done here.
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
        gate_beta: float = 2.0 / 3.0,
        gate_zeta: float = -0.1,
        gate_gamma: float = 1.1,
        causal: bool = False,  # matches model.py's actual GDELTEncoder usage (both blocks non-causal)
    ):
        super().__init__()
        self.input_stack = InputEmbeddingStack(n_categories, n_types, n_subtypes, n_companies, cfg)
        self.hstu1 = HSTUBlock(cfg.embed_dim, n_heads, cfg.dropout, causal=causal)
        self.gate = HardConcreteGate(max_seq_len, gate_beta, gate_zeta, gate_gamma)
        self.hstu2 = HSTUBlock(cfg.embed_dim, n_heads, cfg.dropout, causal=causal)
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(
        self,
        category_ids: torch.Tensor,
        type_ids: torch.Tensor,
        subtype_ids: torch.Tensor,
        company_ids: torch.Tensor,
        position_or_elapsed: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_stack(
            category_ids, type_ids, subtype_ids, company_ids, position_or_elapsed, attention_mask
        )
        x = self.hstu1(x, key_padding_mask=attention_mask)
        x, gates = self.gate(x)
        x = self.hstu2(x, key_padding_mask=attention_mask)
        return self.norm(x), gates

    def l0_loss(self) -> torch.Tensor:
        return self.gate.l0_loss()

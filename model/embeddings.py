"""
Companies House input embedding stack.

Built fresh (not adapted from the GDELT/AML model files, which turned out
to be a different experiment lineage — see working doc discussion). Follows
the architecture locked in across the working doc:

  EventEmbedding      — 3-level hierarchical: category + type + subtype (§7, §9)
                         category uses the §11-corrected enum (partnership-lifecycle
                         split out of incorporation), no intensity/tanh gate (dropped, §CH decisions)
  EntityEmbedding      — company only, no dyad (decided early on)
  DecayedSinusoidalPE  — fixed, not learned, α=1.0 (unchanged from GDELT design)
  SemanticEmbedding    — NOT implemented here; placeholder interface only
                          (frozen sentence transformer → pgvector is a separate,
                          larger task per the working doc's next steps)

Combination pattern follows the same concat-then-project-then-LayerNorm
shape already established in this codebase's other embedding modules
(NeodesicEmbedding): each sub-stream gets its own smaller dim, concatenated,
projected to the full embed_dim. The four top-level streams (event, entity,
PE, semantic) are then summed, per the working doc's stack description.
"""
import math
from typing import Optional

import torch
import torch.nn as nn

from model.config import EmbeddingConfig


class EventEmbedding(nn.Module):
    """
    3-level hierarchical event embedding: category → type → subtype.

    Each level has its own embedding table (different dims — subtype gets
    the most capacity given its ~1,186-way cardinality vs category's ~12).
    Concatenated and projected to embed_dim, following the same pattern as
    this codebase's other multi-stream embeddings.

    No intensity gating (GDELT's tanh-gated intensity doesn't apply here —
    Companies House filings have no analogous confidence/intensity signal,
    per the CH decisions early in the working doc).
    """

    def __init__(self, n_categories: int, n_types: int, n_subtypes: int, cfg: EmbeddingConfig):
        super().__init__()
        self.category_emb = nn.Embedding(n_categories, cfg.category_dim, padding_idx=0)
        self.type_emb = nn.Embedding(n_types, cfg.type_dim, padding_idx=0)
        self.subtype_emb = nn.Embedding(n_subtypes, cfg.subtype_dim, padding_idx=0)

        concat_dim = cfg.category_dim + cfg.type_dim + cfg.subtype_dim
        self.proj = nn.Linear(concat_dim, cfg.embed_dim)
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        category_ids: torch.Tensor,  # [B, L]
        type_ids: torch.Tensor,      # [B, L]
        subtype_ids: torch.Tensor,   # [B, L]
    ) -> torch.Tensor:               # [B, L, D]
        c = self.category_emb(category_ids)
        t = self.type_emb(type_ids)
        s = self.subtype_emb(subtype_ids)
        x = self.proj(torch.cat([c, t, s], dim=-1))
        x = self.norm(x)
        return self.dropout(x)


class EntityEmbedding(nn.Module):
    """
    Company-only entity embedding. No dyad (decided early — CH data is
    single-entity-centric; officer/related-company references feed into
    SemanticEmbedding as text instead, not a structured second entity slot).

    padding_idx=0 mask-zeroes the padding company_id, consistent with how
    padding is handled elsewhere in this codebase's embeddings.
    """

    def __init__(self, n_companies: int, cfg: EmbeddingConfig):
        super().__init__()
        self.company_emb = nn.Embedding(n_companies, cfg.entity_dim, padding_idx=0)
        self.proj = nn.Linear(cfg.entity_dim, cfg.embed_dim)
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, company_ids: torch.Tensor) -> torch.Tensor:  # [B, L] -> [B, L, D]
        x = self.company_emb(company_ids)
        x = self.proj(x)
        return self.norm(x)


class DecayedSinusoidalPE(nn.Module):
    """
    Fixed (not learned) sinusoidal positional encoding, decayed by elapsed
    time since sequence start. α=1.0, matching the GDELT design — this is
    deliberately NOT the per-signal *learned* lambda decay used in the
    AML/population-synthesis models (nd2/neo_cortex); this stays fixed.

    PE(t, 2i)   = sin(t / max_period^(2i/D))
    PE(t, 2i+1) = cos(t / max_period^(2i/D))
    decayed by exp(-α * t / 365) as a recency envelope on the encoding's
    amplitude — distant-past events get a damped positional signal.

    No learnable parameters. div_term is precomputed as a buffer since D
    and max_period are fixed at construction.
    """

    def __init__(self, cfg: EmbeddingConfig):
        super().__init__()
        self.alpha = cfg.pe_alpha
        D = cfg.embed_dim
        div_term = torch.exp(
            torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(cfg.pe_max_period) / D)
        )
        self.register_buffer("div_term", div_term)  # [D/2]
        self.embed_dim = D

    def forward(self, elapsed_days: torch.Tensor) -> torch.Tensor:  # [B, L] -> [B, L, D]
        t = elapsed_days.unsqueeze(-1).float()  # [B, L, 1]
        angles = t * self.div_term  # [B, L, D/2]

        pe = torch.zeros(*elapsed_days.shape, self.embed_dim, device=elapsed_days.device)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)

        decay = torch.exp(-self.alpha * elapsed_days.float() / 365.0).unsqueeze(-1)  # [B, L, 1]
        return pe * decay


class SemanticEmbeddingPlaceholder(nn.Module):
    """
    NOT a real implementation. Returns zeros of the right shape so
    InputEmbeddingStack's forward signature and summation logic are ready
    for the real SemanticEmbedding (frozen sentence transformer -> pgvector
    lookup) without requiring a stack rewrite when it's built.

    Real version will take pre-computed sentence-transformer vectors
    (retrieved from pgvector by event_id) as input, project to embed_dim.
    This placeholder ignores its input entirely.
    """

    def __init__(self, cfg: EmbeddingConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim

    def forward(self, batch_shape: torch.Size, device: torch.device) -> torch.Tensor:
        return torch.zeros(*batch_shape, self.embed_dim, device=device)


class InputEmbeddingStack(nn.Module):
    """
    Combines EventEmbedding + EntityEmbedding + DecayedSinusoidalPE
    (+ optional SemanticEmbedding placeholder), summed — per the working
    doc's stack description. Final LayerNorm + dropout after the sum.
    """

    def __init__(
        self,
        n_categories: int,
        n_types: int,
        n_subtypes: int,
        n_companies: int,
        cfg: EmbeddingConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.event_embedding = EventEmbedding(n_categories, n_types, n_subtypes, cfg)
        self.entity_embedding = EntityEmbedding(n_companies, cfg)
        self.positional_encoding = DecayedSinusoidalPE(cfg)
        self.semantic_embedding = (
            SemanticEmbeddingPlaceholder(cfg) if cfg.include_semantic_placeholder else None
        )
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        category_ids: torch.Tensor,   # [B, L]
        type_ids: torch.Tensor,       # [B, L]
        subtype_ids: torch.Tensor,    # [B, L]
        company_ids: torch.Tensor,    # [B, L]
        elapsed_days: torch.Tensor,   # [B, L]  days since sequence start
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] True = real token
    ) -> torch.Tensor:                # [B, L, D]
        event = self.event_embedding(category_ids, type_ids, subtype_ids)
        entity = self.entity_embedding(company_ids)
        pe = self.positional_encoding(elapsed_days)

        x = event + entity + pe
        if self.semantic_embedding is not None:
            x = x + self.semantic_embedding(category_ids.shape, category_ids.device)

        x = self.final_norm(x)
        x = self.dropout(x)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        return x

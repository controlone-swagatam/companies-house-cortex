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

    Pattern reconciled against GDELT's EventTokenEmbedding (model.py):
    each level gets its own embedding table AND its own projection to
    embed_dim, summed after projection — not concatenated-then-projected.
    (nd2/neo_cortex's NeodesicEmbedding uses concat-then-project instead;
    this follows the actual GDELT reference since that's the lineage this
    architecture descends from.)

    No intensity gating — GDELT's EventTokenEmbedding tanh-gates the cameo
    embedding by Goldstein score; Companies House has no analogous
    confidence/intensity field (dropped per the CH decisions early in the
    working doc), so no equivalent gate here.
    """

    def __init__(self, n_categories: int, n_types: int, n_subtypes: int, cfg: EmbeddingConfig):
        super().__init__()
        self.category_emb = nn.Embedding(n_categories, cfg.category_dim, padding_idx=0)
        self.type_emb = nn.Embedding(n_types, cfg.type_dim, padding_idx=0)
        self.subtype_emb = nn.Embedding(n_subtypes, cfg.subtype_dim, padding_idx=0)

        self.category_proj = nn.Linear(cfg.category_dim, cfg.embed_dim)
        self.type_proj = nn.Linear(cfg.type_dim, cfg.embed_dim)
        self.subtype_proj = nn.Linear(cfg.subtype_dim, cfg.embed_dim)

        self.layer_norm = nn.LayerNorm(cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        category_ids: torch.Tensor,  # [B, L]
        type_ids: torch.Tensor,      # [B, L]
        subtype_ids: torch.Tensor,   # [B, L]
    ) -> torch.Tensor:               # [B, L, D]
        category_emb = self.category_proj(self.category_emb(category_ids))
        type_emb = self.type_proj(self.type_emb(type_ids))
        subtype_emb = self.subtype_proj(self.subtype_emb(subtype_ids))
        return self.dropout(self.layer_norm(category_emb + type_emb + subtype_emb))


class EntityEmbedding(nn.Module):
    """
    Company-only entity embedding. No dyad (decided early — CH data is
    single-entity-centric; officer/related-company references feed into
    SemanticEmbedding as text instead, not a structured second entity slot).

    Reconciled against GDELT's CountryEmbedding: sequence-level, not
    per-token. A training sequence is one company's filing history, so
    company_id is constant across the whole sequence — it should be looked
    up once per sequence and broadcast-added, exactly like GDELT's country
    context, not embedded per-token with a [B, L] id tensor (the original
    version of this module did that, which was both wasteful and
    semantically wrong — company doesn't vary within a sequence).

    No LayerNorm/dropout, matching CountryEmbedding exactly.
    """

    def __init__(self, n_companies: int, cfg: EmbeddingConfig):
        super().__init__()
        self.company_emb = nn.Embedding(n_companies, cfg.entity_dim, padding_idx=0)
        self.proj = nn.Linear(cfg.entity_dim, cfg.embed_dim)

    def forward(self, company_ids: torch.Tensor) -> torch.Tensor:
        # company_ids: [B]  (one id per sequence, NOT [B, L])
        # returns: [B, D] — caller broadcasts via unsqueeze(1), same as
        # GDELTEncoder does with country_embed
        return self.proj(self.company_emb(company_ids))


class LearnedPositionalEmbedding(nn.Module):
    """
    Plain learned absolute positional embedding — matches model.py's actual
    GDELT implementation (nn.Embedding(max_seq_len, d_model)) exactly.
    """

    def __init__(self, cfg: EmbeddingConfig):
        super().__init__()
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:  # [B, L] int -> [B, L, D]
        return self.pos_emb(positions)


class DecayedSinusoidalPE(nn.Module):
    """
    Fixed (not learned) sinusoidal positional encoding, decayed by elapsed
    time since sequence start. α=1.0.

    NOTE: this is the "canonical" design per the working doc, but model.py
    (the actual GDELT reference code) uses LearnedPositionalEmbedding
    instead — see EmbeddingConfig.positional_encoding_type. Kept here as an
    option since decayed sinusoidal PE has a real rationale for CH's
    irregular filing intervals, but it's not what's currently deployed for
    GDELT, so don't assume this is the active default without checking the
    config.

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
    Combines EventEmbedding + EntityEmbedding + positional encoding
    (+ optional SemanticEmbedding placeholder). Reconciled against
    GDELTEncoder.forward's actual pattern:

        x = event_embed(...)
        x = x + pos_embed(positions)              # per-token, broadcast over batch
        x = x + country_embed(country_ids).unsqueeze(1)  # per-sequence, broadcast over length

    i.e. entity embedding is added once per sequence and broadcast across
    all positions, not computed per-token — see EntityEmbedding's docstring.
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

        if cfg.positional_encoding_type == "learned":
            self.positional_encoding = LearnedPositionalEmbedding(cfg)
        elif cfg.positional_encoding_type == "decayed_sinusoidal":
            self.positional_encoding = DecayedSinusoidalPE(cfg)
        else:
            raise ValueError(
                f"Unknown positional_encoding_type: {cfg.positional_encoding_type!r} "
                "(expected 'learned' or 'decayed_sinusoidal')"
            )

        self.semantic_embedding = (
            SemanticEmbeddingPlaceholder(cfg) if cfg.include_semantic_placeholder else None
        )

    def forward(
        self,
        category_ids: torch.Tensor,   # [B, L]
        type_ids: torch.Tensor,       # [B, L]
        subtype_ids: torch.Tensor,    # [B, L]
        company_ids: torch.Tensor,    # [B]      — one per SEQUENCE, not per token
        position_or_elapsed: torch.Tensor,  # [B, L] — token index (learned PE) or elapsed days (decayed sinusoidal)
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] True = real token
    ) -> torch.Tensor:                # [B, L, D]
        x = self.event_embedding(category_ids, type_ids, subtype_ids)
        x = x + self.positional_encoding(position_or_elapsed)
        x = x + self.entity_embedding(company_ids).unsqueeze(1)  # broadcast [B, D] -> [B, L, D]

        if self.semantic_embedding is not None:
            x = x + self.semantic_embedding(category_ids.shape, category_ids.device)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        return x

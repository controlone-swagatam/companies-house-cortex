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
    Fixed (not learned) sinusoidal positional encoding, decayed so the MOST
    RECENT real event in each sequence gets the strongest signal and older
    events decay away — per CORTEX_ARCHITECTURE.md §2.1 ("newest position
    2.7x stronger than oldest", alpha=1.0 default).

    IMPORTANT DIRECTIONALITY NOTE: the canonical formula PE(pos,2i) =
    exp(-alpha*pos/N)*sin(pos/...) decays with INCREASING pos. Taken
    literally with pos = standard forward-chronological index (0=oldest,
    L-1=newest, the ordering causal masking requires — "i cannot attend to
    j>i" only makes sense chronologically forward), that would make the
    OLDEST event strongest and decay TOWARD the newest — the opposite of
    what the doc claims in prose. The only self-consistent reading is that
    the doc's "pos" means "steps back from the most recent real event",
    not raw forward index. That's what's implemented here: pos_for_pe =
    (real_length - 1) - forward_position, computed per-sequence using
    attention_mask so padding doesn't corrupt which position counts as
    "most recent". Forward-chronological indexing is preserved for
    everything else (embeddings, causal masking) — only the PE's internal
    phase/decay variable is reversed.
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

    def forward(
        self,
        positions: torch.Tensor,  # [B, L] standard forward-chronological indices (0=oldest)
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] bool, True = real token
    ) -> torch.Tensor:  # [B, L, D]
        if attention_mask is not None:
            real_lengths = attention_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
        else:
            real_lengths = torch.full(
                (positions.shape[0], 1), float(positions.shape[1]), device=positions.device
            )

        # "steps back from the most recent real position" — 0 at the last
        # real event, increasing going further into the past.
        steps_from_recent = (real_lengths - 1) - positions.float()  # [B, L]

        angles = steps_from_recent.unsqueeze(-1) * self.div_term  # [B, L, D/2]
        pe = torch.zeros(*positions.shape, self.embed_dim, device=positions.device)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)

        # Normalization constant N = each sequence's OWN (real_length - 1),
        # not a fixed global max_seq_len. This is what makes "newest 2.7x
        # stronger than oldest" (alpha=1.0 -> ratio = e^alpha ~ 2.718) hold
        # universally regardless of how long a given sequence actually is —
        # the oldest position in ANY sequence is always exactly alpha
        # e-foldings old relative to that sequence's own span. Using a fixed
        # max_seq_len instead (tried first, verified wrong in testing) made
        # short sequences barely decay at all relative to a much larger
        # constant, giving a ~1.15x ratio instead of the intended ~2.7x.
        norm_const = (real_lengths - 1).clamp(min=1)  # [B, 1], avoid div-by-zero for length-1
        decay = torch.exp(-self.alpha * steps_from_recent.clamp(min=0) / norm_const).unsqueeze(-1)
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
        if self.cfg.positional_encoding_type == "decayed_sinusoidal":
            x = x + self.positional_encoding(position_or_elapsed, attention_mask)
        else:
            x = x + self.positional_encoding(position_or_elapsed)
        x = x + self.entity_embedding(company_ids).unsqueeze(1)  # broadcast [B, D] -> [B, L, D]

        if self.semantic_embedding is not None:
            x = x + self.semantic_embedding(category_ids.shape, category_ids.device)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        return x

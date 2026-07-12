"""
Config for the Companies House EventEmbedding / EntityEmbedding /
DecayedSinusoidalPE stack. Separate from the training-hyperparameter
config referenced elsewhere (GDELT's ntst_config.py) since this is
specific to the input embedding layer built here.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    embed_dim: int = 128

    # Per-level embedding dims within EventEmbedding, summed to embed_dim
    # after projection. Category gets the smallest share (fewest classes,
    # ~12 post-§11), subtype the largest (~1,186-way, needs more capacity
    # to differentiate that many classes) — type sits between them.
    category_dim: int = 16
    type_dim: int = 32
    subtype_dim: int = 64

    entity_dim: int = 16  # company-only, no dyad (per earlier decision) — small, single lookup

    # DecayedSinusoidalPE vs plain learned positional embedding.
    # RESOLVED: CORTEX_ARCHITECTURE.md (the canonical spec, confirmed against
    # model_v2.py — the current working GDELT implementation) confirms
    # DecayedSinusoidalPE (fixed, alpha=1.0) is correct. model.py (used for
    # the earlier reconciliation) was the STALE v1 file; 'learned' is kept
    # available for compatibility but is no longer the default.
    positional_encoding_type: str = "decayed_sinusoidal"  # "decayed_sinusoidal" | "learned"
    max_seq_len: int = 512  # only used when positional_encoding_type == "learned"
    pe_alpha: float = 1.0  # fixed, not learned
    pe_max_period: float = 10000.0

    dropout: float = 0.1

    # SemanticEmbedding is NOT built here — this is a placeholder dim only,
    # so InputEmbeddingStack's projection layer has the right shape ready
    # for when it's implemented (frozen sentence transformer -> pgvector,
    # per the working doc's next-steps list). Real semantic vectors from a
    # sentence-transformer are typically 384 (MiniLM-class), used here as
    # the placeholder default.
    semantic_dim: int = 384
    include_semantic_placeholder: bool = False  # off by default until SemanticEmbedding exists

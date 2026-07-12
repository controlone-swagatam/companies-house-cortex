"""
Full Companies House model: causal-MLM encoder (3 HSTU blocks, 2 gates,
per CORTEX_ARCHITECTURE.md) + hierarchical prediction heads.

Training paradigm (decided earlier, still applies): causal MLM — mask
random events, predict them, causal attention throughout so no future
leakage (§9). This is fully compatible with the canonical spec's own
causal-mask requirement.

Head design note: the canonical spec's §2.4 MLM Head is a single flat
Linear(hidden_dim, vocab_size) — appropriate for GDELT's flatter CAMEO
vocab. Companies House keeps the hierarchical category->type->subtype
heads from §7 of the working doc instead, a deliberate adaptation
justified by the ~1,186-way, heavily skewed subtype vocabulary found in
§6/§10 — a flat head over that vocab would be data-inefficient in a way
GDELT's vocab doesn't suffer from. This is a documented deviation, not an
oversight.
"""
from typing import Optional

import torch
import torch.nn as nn

from model.config import EmbeddingConfig
from model.encoder import CompaniesHouseEncoder
from model.heads import HierarchicalPredictionHead, hierarchical_mlm_loss
from model.masking import apply_mlm_masking
from model.vocab import EventVocab


class CompaniesHouseModel(nn.Module):
    def __init__(
        self,
        vocab: EventVocab,
        cfg: EmbeddingConfig,
        n_heads: int = 4,
        max_seq_len: int = 512,
        head_hidden_dim: int = 128,
        mask_prob: float = 0.15,
        min_context: int = 1,  # CH default, NOT GDELT's 10 — see note below
        gate1_target: float = 0.15,
        gate2_target: float = 0.05,
    ):
        """
        min_context default deviates from CORTEX_ARCHITECTURE.md's GDELT
        default of 10. Measured against real CH data: median sequence
        length is 3 events, mean 4.3 — at min_context=10, only 3.1% of
        companies (92/3000) would ever have a single eligible masking
        position, wasting ~97% of the training data. GDELT's country-pair
        dyads span years of accumulated events; a single company's filing
        history typically doesn't. min_context=1 lets the shortest viable
        sequences (length 2, the dataset's own floor) still contribute
        exactly one masked position.
        """
        super().__init__()
        self.vocab = vocab
        self.mask_prob = mask_prob
        self.min_context = min_context

        self.encoder = CompaniesHouseEncoder(
            n_categories=vocab.n_categories,
            n_types=vocab.n_types,
            n_subtypes=vocab.n_subtypes,
            n_companies=vocab.n_companies,
            cfg=cfg,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            gate1_target=gate1_target,
            gate2_target=gate2_target,
        )
        self.heads = HierarchicalPredictionHead(
            d_model=cfg.embed_dim,
            n_categories=vocab.n_categories,
            n_types=vocab.n_types,
            n_subtypes=vocab.n_subtypes,
            hidden_dim=head_hidden_dim,
        )

    def forward_training(
        self,
        category_ids: torch.Tensor,
        type_ids: torch.Tensor,
        subtype_ids: torch.Tensor,
        company_ids: torch.Tensor,
        position_or_elapsed: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        masked = apply_mlm_masking(
            category_ids, type_ids, subtype_ids, attention_mask,
            category_mask_id=self.vocab.category_mask_id,
            type_mask_id=self.vocab.type_mask_id,
            subtype_mask_id=self.vocab.subtype_mask_id,
            n_categories=self.vocab.n_categories,
            n_types=self.vocab.n_types,
            n_subtypes=self.vocab.n_subtypes,
            mask_prob=self.mask_prob,
            min_context=self.min_context,
        )

        enc_out = self.encoder(
            masked.category_ids, masked.type_ids, masked.subtype_ids,
            company_ids, position_or_elapsed, attention_mask,
        )
        category_logits, type_logits, subtype_logits = self.heads(enc_out["hidden"])

        losses = hierarchical_mlm_loss(
            category_logits, type_logits, subtype_logits,
            masked.category_labels, masked.type_labels, masked.subtype_labels,
        )

        return {
            **losses,
            "gates1": enc_out["gates1"],
            "gates2": enc_out["gates2"],
            "mask_positions": masked.mask_positions,
            "category_logits": category_logits,
            "type_logits": type_logits,
            "subtype_logits": subtype_logits,
        }

    def forward_inference(
        self,
        category_ids: torch.Tensor,
        type_ids: torch.Tensor,
        subtype_ids: torch.Tensor,
        company_ids: torch.Tensor,
        position_or_elapsed: torch.Tensor,
        attention_mask: torch.Tensor,
        predict_position: Optional[int] = None,
        return_attn: bool = False,
    ) -> dict:
        """
        No masking except at the target position — for the "what happens
        next" use case, mask the last real position (or a caller-specified
        one) and predict it. Causal attention means this only sees
        genuinely earlier events, matching training.
        """
        B, L = category_ids.shape
        if predict_position is None:
            lengths = attention_mask.sum(dim=1) - 1
        else:
            lengths = torch.full((B,), predict_position, device=category_ids.device)

        category_in = category_ids.clone()
        type_in = type_ids.clone()
        subtype_in = subtype_ids.clone()

        batch_idx = torch.arange(B, device=category_ids.device)
        category_in[batch_idx, lengths] = self.vocab.category_mask_id
        type_in[batch_idx, lengths] = self.vocab.type_mask_id
        subtype_in[batch_idx, lengths] = self.vocab.subtype_mask_id

        self.eval()
        with torch.no_grad():
            enc_out = self.encoder(
                category_in, type_in, subtype_in, company_ids, position_or_elapsed,
                attention_mask, return_attn=return_attn,
            )
            category_logits, type_logits, subtype_logits = self.heads(enc_out["hidden"])

        result = {
            "category_logits": category_logits[batch_idx, lengths],
            "type_logits": type_logits[batch_idx, lengths],
            "subtype_logits": subtype_logits[batch_idx, lengths],
            "predicted_positions": lengths,
            "gates1": enc_out["gates1"],
            "gates2": enc_out["gates2"],
        }
        if return_attn:
            result["attn1"] = enc_out["attn1"]
            result["attn2"] = enc_out["attn2"]
            result["attn3"] = enc_out["attn3"]
        return result

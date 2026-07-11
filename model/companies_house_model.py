"""
Full Companies House model: causal-MLM encoder + hierarchical prediction
heads. Ties together model/encoder.py, model/masking.py, model/heads.py.

Training paradigm (decided): MLM-style masking with a CAUSAL attention
mask — mask random events, predict them, but each masked position can
only attend to genuinely earlier events in the sequence. This is stricter
than standard BERT (which is fully bidirectional) but consistent with
§9's no-future-leakage principle, and closer to what a real inference-time
"predict what's coming" use case needs than plain bidirectional MLM would be.
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
    ):
        super().__init__()
        self.vocab = vocab
        self.mask_prob = mask_prob

        self.encoder = CompaniesHouseEncoder(
            n_categories=vocab.n_categories,
            n_types=vocab.n_types,
            n_subtypes=vocab.n_subtypes,
            n_companies=vocab.n_companies,
            cfg=cfg,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            causal=True,  # decided: causal MLM, not bidirectional
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
        """
        Applies MLM masking, runs the causal encoder + heads, returns losses
        and predictions. Use this for training steps.
        """
        masked = apply_mlm_masking(
            category_ids, type_ids, subtype_ids, attention_mask,
            category_mask_id=self.vocab.category_mask_id,
            type_mask_id=self.vocab.type_mask_id,
            subtype_mask_id=self.vocab.subtype_mask_id,
            n_categories=self.vocab.n_categories,
            n_types=self.vocab.n_types,
            n_subtypes=self.vocab.n_subtypes,
            mask_prob=self.mask_prob,
        )

        hidden, gates = self.encoder(
            masked.category_ids, masked.type_ids, masked.subtype_ids,
            company_ids, position_or_elapsed, attention_mask,
        )
        category_logits, type_logits, subtype_logits = self.heads(hidden)

        losses = hierarchical_mlm_loss(
            category_logits, type_logits, subtype_logits,
            masked.category_labels, masked.type_labels, masked.subtype_labels,
        )
        losses["l0_loss"] = self.encoder.l0_loss()

        return {
            **losses,
            "gates": gates,
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
    ) -> dict:
        """
        No masking — for inference, mask the LAST real position explicitly
        (or a caller-specified position) and predict it, since that's the
        "what happens next" use case. Causal attention means this
        prediction only sees genuinely earlier events, matching training.
        """
        B, L = category_ids.shape
        if predict_position is None:
            # last real (non-pad) position per sequence
            lengths = attention_mask.sum(dim=1) - 1  # [B]
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
            hidden, gates = self.encoder(
                category_in, type_in, subtype_in, company_ids, position_or_elapsed, attention_mask
            )
            category_logits, type_logits, subtype_logits = self.heads(hidden)

        return {
            "category_logits": category_logits[batch_idx, lengths],  # [B, n_categories]
            "type_logits": type_logits[batch_idx, lengths],
            "subtype_logits": subtype_logits[batch_idx, lengths],
            "predicted_positions": lengths,
            "gates": gates,
        }

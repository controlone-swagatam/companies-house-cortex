"""
Masking for causal training, per CORTEX_ARCHITECTURE.md (confirmed spec,
not the BERT-style random masking used in the earlier draft):

  - Only positions with >= min_context (default 10) real predecessors are
    eligible for masking — a masked position needs enough antecedent
    context to make prediction meaningful, not just "not padding".
  - Mask the LAST mlm_mask_frac (default 0.15) fraction of each sequence's
    eligible positions — i.e. a contiguous suffix ending at the sequence's
    last real event, not scattered random positions. This is a much
    closer fit to "predict what happens next" than classic BERT MLM: the
    model is trained specifically to predict the most recent stretch of
    events given everything before it.
  - Masked positions are replaced with the MASK token 100% of the time —
    no BERT-style 80/10/10 mask/random/keep mixture, since that mixture
    isn't part of the confirmed spec and isn't obviously motivated here.

Combined with causal attention, a masked position within the suffix block
can still attend to EARLIER masked positions in that same block (only
future positions are blocked) — this is intentional, not a leak: it's the
standard behaviour for this kind of infilling objective, and those earlier
masked positions carry no information beyond "something was masked here"
since their embeddings are the MASK token, not their true values.
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class MaskedBatch:
    category_ids: torch.Tensor
    type_ids: torch.Tensor
    subtype_ids: torch.Tensor
    category_labels: torch.Tensor  # original ids at masked positions, -100 elsewhere
    type_labels: torch.Tensor
    subtype_labels: torch.Tensor
    mask_positions: torch.Tensor   # [B, L] bool


def apply_mlm_masking(
    category_ids: torch.Tensor,
    type_ids: torch.Tensor,
    subtype_ids: torch.Tensor,
    attention_mask: torch.Tensor,  # [B, L] bool, True = real (non-pad) token
    category_mask_id: int,
    type_mask_id: int,
    subtype_mask_id: int,
    n_categories: int,   # unused now (kept for call-site compatibility — no random-token corruption anymore)
    n_types: int,
    n_subtypes: int,
    mask_prob: float = 0.15,   # mlm_mask_frac in the canonical naming
    min_context: int = 1,  # CH default (see companies_house_model.py) — NOT GDELT's 10
) -> MaskedBatch:
    B, L = category_ids.shape
    device = category_ids.device

    mask_positions = torch.zeros(B, L, dtype=torch.bool, device=device)

    for b in range(B):
        real_length = int(attention_mask[b].sum().item())
        n_eligible = max(0, real_length - min_context)
        n_to_mask = math.ceil(mask_prob * n_eligible)
        if n_to_mask > 0:
            # last n_to_mask positions of the real sequence (indices
            # real_length - n_to_mask .. real_length - 1)
            mask_positions[b, real_length - n_to_mask: real_length] = True

    category_labels = torch.where(mask_positions, category_ids, torch.full_like(category_ids, -100))
    type_labels = torch.where(mask_positions, type_ids, torch.full_like(type_ids, -100))
    subtype_labels = torch.where(mask_positions, subtype_ids, torch.full_like(subtype_ids, -100))

    category_out = category_ids.clone()
    type_out = type_ids.clone()
    subtype_out = subtype_ids.clone()
    category_out[mask_positions] = category_mask_id
    type_out[mask_positions] = type_mask_id
    subtype_out[mask_positions] = subtype_mask_id

    return MaskedBatch(
        category_ids=category_out,
        type_ids=type_out,
        subtype_ids=subtype_out,
        category_labels=category_labels,
        type_labels=type_labels,
        subtype_labels=subtype_labels,
        mask_positions=mask_positions,
    )

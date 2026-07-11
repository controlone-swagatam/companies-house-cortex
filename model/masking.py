"""
MLM-style masking for causal training.

Standard BERT-style masking (~15% of positions, mask/random/keep split)
applied to category_ids/type_ids/subtype_ids jointly — a masked *event*
means all three levels at that position are masked together, since they
describe the same underlying filing (doesn't make sense to mask category
but not type for the same event).

Combined with causal attention (HSTUBlock(causal=True)) in the encoder,
so a masked position's prediction is conditioned only on genuinely earlier
events in the sequence — consistent with §9's "date, not action_date"
no-future-leakage principle. This is stricter than standard BERT (which
uses full bidirectional context even for masked positions) but matches
what we actually need: a model usable for real next-event-style inference,
not just an offline pretraining objective.

PAD positions (attention_mask == False) are never eligible for masking.
"""
from dataclasses import dataclass

import torch


@dataclass
class MaskedBatch:
    category_ids: torch.Tensor    # masked input ids
    type_ids: torch.Tensor
    subtype_ids: torch.Tensor
    category_labels: torch.Tensor  # original ids at masked positions, -100 elsewhere (CE ignore_index)
    type_labels: torch.Tensor
    subtype_labels: torch.Tensor
    mask_positions: torch.Tensor   # [B, L] bool — which positions were masked


def apply_mlm_masking(
    category_ids: torch.Tensor,
    type_ids: torch.Tensor,
    subtype_ids: torch.Tensor,
    attention_mask: torch.Tensor,  # [B, L] bool, True = real (non-pad) token
    category_mask_id: int,
    type_mask_id: int,
    subtype_mask_id: int,
    n_categories: int,
    n_types: int,
    n_subtypes: int,
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,   # of masked positions: 80% -> MASK token
    random_token_prob: float = 0.1,  # 10% -> random token, 10% -> unchanged (standard BERT split)
) -> MaskedBatch:
    device = category_ids.device
    shape = category_ids.shape

    # Eligible = real tokens only (never mask padding)
    eligible = attention_mask.clone()
    rand = torch.rand(shape, device=device)
    mask_positions = eligible & (rand < mask_prob)

    # Labels: original id at masked positions, -100 (CE ignore_index) elsewhere
    category_labels = torch.where(mask_positions, category_ids, torch.full_like(category_ids, -100))
    type_labels = torch.where(mask_positions, type_ids, torch.full_like(type_ids, -100))
    subtype_labels = torch.where(mask_positions, subtype_ids, torch.full_like(subtype_ids, -100))

    # 80/10/10 split within masked positions
    action_rand = torch.rand(shape, device=device)
    use_mask_token = mask_positions & (action_rand < mask_token_prob)
    use_random_token = mask_positions & (action_rand >= mask_token_prob) & (
        action_rand < mask_token_prob + random_token_prob
    )
    # remaining masked positions (~10%) keep original value — no action needed

    category_out = category_ids.clone()
    type_out = type_ids.clone()
    subtype_out = subtype_ids.clone()

    category_out[use_mask_token] = category_mask_id
    type_out[use_mask_token] = type_mask_id
    subtype_out[use_mask_token] = subtype_mask_id

    if use_random_token.any():
        n_random = int(use_random_token.sum().item())
        category_out[use_random_token] = torch.randint(2, n_categories, (n_random,), device=device)
        type_out[use_random_token] = torch.randint(2, n_types, (n_random,), device=device)
        subtype_out[use_random_token] = torch.randint(2, n_subtypes, (n_random,), device=device)

    return MaskedBatch(
        category_ids=category_out,
        type_ids=type_out,
        subtype_ids=subtype_out,
        category_labels=category_labels,
        type_labels=type_labels,
        subtype_labels=subtype_labels,
        mask_positions=mask_positions,
    )

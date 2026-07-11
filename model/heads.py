"""
Hierarchical prediction heads, per §7 of the working doc.

Three heads, each conditioned on the one above:
    category head:  hidden -> n_categories
    type head:       [hidden ; category_logits/probs] -> n_types
    subtype head:     [hidden ; type_logits/probs] -> n_subtypes

Conditioning is implemented by concatenating the *soft* distribution
(softmax probs, not hard argmax) from the level above into the next head's
input — differentiable end-to-end, and doesn't force a hard commitment to
a possibly-wrong category prediction during training. At inference, the
caller can still take argmax per level if a hard prediction is wanted.

This is the resolution to §7's "predict hierarchically" design: rather
than one flat ~1,186-way softmax over subtypes (data-inefficient given the
long-tail skew found in §6/§10), each level's classification problem is
conditioned on (and much easier given) the level above.
"""
import torch
import torch.nn as nn


class HierarchicalPredictionHead(nn.Module):
    def __init__(self, d_model: int, n_categories: int, n_types: int, n_subtypes: int, hidden_dim: int = 128):
        super().__init__()
        self.category_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_categories),
        )
        self.type_head = nn.Sequential(
            nn.Linear(d_model + n_categories, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_types),
        )
        self.subtype_head = nn.Sequential(
            nn.Linear(d_model + n_types, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_subtypes),
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden: [B, L, D]
        category_logits = self.category_head(hidden)               # [B, L, n_categories]
        category_probs = torch.softmax(category_logits, dim=-1)

        type_input = torch.cat([hidden, category_probs], dim=-1)
        type_logits = self.type_head(type_input)                    # [B, L, n_types]
        type_probs = torch.softmax(type_logits, dim=-1)

        subtype_input = torch.cat([hidden, type_probs], dim=-1)
        subtype_logits = self.subtype_head(subtype_input)           # [B, L, n_subtypes]

        return category_logits, type_logits, subtype_logits


def hierarchical_mlm_loss(
    category_logits: torch.Tensor,
    type_logits: torch.Tensor,
    subtype_logits: torch.Tensor,
    category_labels: torch.Tensor,  # -100 = ignore (unmasked positions)
    type_labels: torch.Tensor,
    subtype_labels: torch.Tensor,
) -> dict:
    """
    Independent cross-entropy per level at masked positions (ignore_index=-100
    handles unmasked positions automatically). Returns per-level losses plus
    the summed total, so a caller can log/weight them separately if the
    level losses turn out to need different weighting (e.g. if subtype loss
    dominates given its much larger vocab).
    """
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    category_loss = ce(category_logits.reshape(-1, category_logits.size(-1)), category_labels.reshape(-1))
    type_loss = ce(type_logits.reshape(-1, type_logits.size(-1)), type_labels.reshape(-1))
    subtype_loss = ce(subtype_logits.reshape(-1, subtype_logits.size(-1)), subtype_labels.reshape(-1))

    return {
        "category_loss": category_loss,
        "type_loss": type_loss,
        "subtype_loss": subtype_loss,
        "total_loss": category_loss + type_loss + subtype_loss,
    }

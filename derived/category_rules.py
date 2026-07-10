"""
Category correction rules for the derived layer.

This is the business logic explicitly kept OUT of the receptor
(common/ch_client.py, inference/api_receptor.py, training/batch_ingest.py
all pass through CH's raw `category` field untouched — see their
docstrings). This module is where category corrections like §11's LP fix
belong instead.

Design: a small ordered list of rules, each a (predicate, corrected_category)
pair. Kept as explicit, inspectable rules rather than a single hardcoded
if/else, so a second correction (found the same way — real-sample
inspection) can be added without touching the core logic, only the rule
table.
"""
from typing import Callable, NamedTuple, Optional


class CategoryRule(NamedTuple):
    name: str                                    # short identifier, for logging/audit
    predicate: Callable[[dict], bool]             # given a raw filingHistoryItem, does this rule apply?
    corrected_category: str
    reason: str                                   # why this override exists, for anyone reading the rule table


# §11: Limited Partnership lifecycle forms (LP5/LP6/LP7) are filed under
# CH's `incorporation` category regardless of whether they're a one-time
# registration (LP5, LP7) or a recurring change notice (LP6). LP6 alone
# breaks the "incorporation happens once" assumption once LPs are in the
# training population — observed up to 7x for a single company in a
# 200-company real sample.
_LP_LIFECYCLE_TYPES = {"LP5", "LP6", "LP7"}

CATEGORY_RULES: list[CategoryRule] = [
    CategoryRule(
        name="lp_lifecycle_reclassification",
        predicate=lambda item: (
            item.get("category") == "incorporation"
            and item.get("type") in _LP_LIFECYCLE_TYPES
        ),
        corrected_category="partnership-lifecycle",
        reason=(
            "CH files LP5/LP6/LP7 under 'incorporation' regardless of "
            "recurrence. LP6 is a recurring change notice, not a one-time "
            "incorporation event — see working doc §11."
        ),
    ),
]


def correct_category(raw_item: dict) -> tuple[str, Optional[str]]:
    """
    Returns (corrected_category, applied_rule_name). applied_rule_name is
    None if no rule matched, i.e. CH's original category is used as-is.
    """
    original = raw_item.get("category")
    for rule in CATEGORY_RULES:
        if rule.predicate(raw_item):
            return rule.corrected_category, rule.name
    return original, None

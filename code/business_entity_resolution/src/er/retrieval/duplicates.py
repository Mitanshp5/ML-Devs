"""Duplicate target expansion used by the final test-cache builder."""
from __future__ import annotations

import pandas as pd


def build_duplicate_map(pool_df: pd.DataFrame) -> dict[str, list[str]]:
    """Map each target to the non-empty exact-text duplicates in its pool."""
    valid = pool_df[
        (pool_df.business_name.str.strip() != "")
        & (pool_df.business_address.str.strip() != "")
    ]
    groups = valid.groupby(["business_name", "business_address"])["entity_id"].apply(list)
    result: dict[str, list[str]] = {}
    for ids in groups[groups.apply(len) > 1]:
        for entity_id in ids:
            result[entity_id] = ids
    return result

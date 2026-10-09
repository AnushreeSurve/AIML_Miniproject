"""Apriori association rules, mined separately within each arm (exploratory).

Each patient becomes a "basket" of items such as ``X_va_letters_band=<=54``
or ``X_insulin=1``, plus the outcome item ``gain>=15``. *Apriori* finds item
sets that occur in at least ``min_support`` of baskets, using the fact that
every subset of a frequent set must itself be frequent. From these we keep
rules ``{items} -> {gain>=15}`` and rank them by *lift* = confidence /
P(gain>=15): lift > 1 means the items go with more responders than average
*in that arm*. Rules describe co-occurrence in this sample; they are not
causal and not evidence that a drug works better for that group.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

OUTCOME_ITEM = "gain>=15"


def baskets(df: pd.DataFrame, bands: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Boolean one-hot item matrix: band items, binary items and the outcome item."""
    items = pd.get_dummies(bands.astype(object), prefix_sep="=", dtype=bool)
    for col in cfg["represent"]["apriori"]["binary_items"]:
        items[f"{col}=1"] = df[col].fillna(0).astype(int).eq(1).to_numpy()
    thr = cfg["analysis"]["responder_threshold_letters"]
    items[OUTCOME_ITEM] = (df[cfg["analysis"]["primary_outcome"]] >= thr).to_numpy()
    return items.reset_index(drop=True)


def mine_rules(items: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Rules with the outcome item as the only consequent, sorted by lift."""
    a = cfg["represent"]["apriori"]
    freq = apriori(items, min_support=a["min_support"], use_colnames=True, max_len=a["max_len"])
    if freq.empty:
        return pd.DataFrame()
    rules = association_rules(freq, metric="confidence", min_threshold=a["min_confidence"])
    rules = rules[rules["consequents"].apply(lambda c: c == frozenset({OUTCOME_ITEM}))]
    rules = rules[rules["antecedents"].apply(lambda c: OUTCOME_ITEM not in c)]
    out = pd.DataFrame(
        {
            "antecedents": rules["antecedents"].apply(lambda s: " & ".join(sorted(s))),
            "consequent": OUTCOME_ITEM,
            "support": rules["support"],
            "confidence": rules["confidence"],
            "lift": rules["lift"],
        }
    )
    return out.sort_values(["lift", "support"], ascending=False).reset_index(drop=True)


def rules_by_arm(df: pd.DataFrame, bands: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Top rules by lift within each arm (patients with an observed outcome only)."""
    observed = df[cfg["analysis"]["primary_outcome"]].notna().to_numpy()
    top = cfg["represent"]["apriori"]["top_n"]
    out = []
    for arm in cfg["analysis"]["arms"]:
        mask = observed & (df["treatment"] == arm).to_numpy()
        items = baskets(df[mask], bands[mask], cfg)
        r = mine_rules(items, cfg).head(top)
        r.insert(0, "arm", arm)
        r.insert(1, "n_baskets", int(mask.sum()))
        r.insert(2, "base_rate_gain15", float(items[OUTCOME_ITEM].mean()))
        out.append(r)
    res = pd.concat(out, ignore_index=True)
    res["note"] = "exploratory co-occurrence - not causal"
    return res

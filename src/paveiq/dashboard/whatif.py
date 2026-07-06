"""Hypothetical feature-change simulation for the what-if panel.

No ``streamlit`` import — pure pandas, testable without a browser.
The caller (``app.py``) is responsible for calling the *same*
``registry.load_scorer(...).predict()`` on both the original and
modified subsets, so what-if and production scoring can never drift
apart:

    original = select_scope(df, scope, target)
    modified, affected = apply_hypothetical(df, toggles, scope, target)
    before = scorer.predict(original)
    after = scorer.predict(modified)

Four toggles, one per scorer input column — the natural complete set
given the feature schema:

====================  ==================  ============================
Toggle                Column changed      Typical target
====================  ==================  ============================
``surface_upgrade``   ``surface_quality`` 0.6 (compacted) or 1.0 (paved)
``add_sidewalk``      ``sidewalk_presence`` ``"explicit_present"``
``widen_path``        ``width_m``         a user-chosen metres value
``pedestrianize``     ``highway_likelihood`` 1.0
====================  ==================  ============================

When a ward is the scope, the change is bulk-applied to every
matching segment in that ward (e.g. every unpaved segment becomes
paved) — not just a representative sample.
"""

from __future__ import annotations

import pandas as pd

TOGGLE_COLUMNS = {
    "surface_upgrade": "surface_quality",
    "add_sidewalk": "sidewalk_presence",
    "widen_path": "width_m",
    "pedestrianize": "highway_likelihood",
}


def select_scope(df: pd.DataFrame, scope: str, target) -> pd.DataFrame:
    """Return the rows in ``df`` matching ``scope``/``target``.

    ``scope`` is ``"segment"`` (``target`` is an ``osmid``) or
    ``"ward"`` (``target`` is a ``ward_id``; bulk-selects every
    segment in that ward).
    """
    if scope == "segment":
        mask = df["osmid"] == target
    elif scope == "ward":
        mask = df["ward_id"] == target
    else:
        raise ValueError(f"scope must be 'segment' or 'ward', got {scope!r}")
    return df.loc[mask]


def apply_hypothetical(
    df: pd.DataFrame,
    toggles: dict,
    scope: str,
    target,
) -> tuple[pd.DataFrame, dict]:
    """Apply ``toggles`` to the rows selected by ``scope``/``target``.

    ``toggles`` maps toggle name (a key of ``TOGGLE_COLUMNS``) to the
    target value for that column, e.g.
    ``{"surface_upgrade": 1.0, "add_sidewalk": "explicit_present"}``.
    Only toggles present in the dict are applied; within those, only
    rows whose current value differs from the target actually change
    (an already-paved segment is untouched by ``surface_upgrade``).

    Returns the modified subset (same index as the matching rows in
    ``df``) and a ``{toggle_name: n_affected}`` dict so the caller can
    show "N of M segments affected" per toggle.
    """
    unknown = set(toggles) - set(TOGGLE_COLUMNS)
    if unknown:
        raise ValueError(f"unknown toggle(s): {sorted(unknown)}")

    subset = select_scope(df, scope, target).copy()
    affected = {}
    for toggle_name, target_value in toggles.items():
        col = TOGGLE_COLUMNS[toggle_name]
        differs = subset[col] != target_value
        affected[toggle_name] = int(differs.sum())
        subset.loc[differs, col] = target_value
    return subset, affected

"""PaveIQ Streamlit dashboard.

Thin UI glue only — every function with real logic lives in
``data.py``/``mapview.py``/``whatif.py`` and is unit-tested there.
This module wires them to Streamlit widgets across three tabs: a 3D
map, a ward-level leaderboard, and a what-if simulator.

Run with::

    streamlit run src/paveiq/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from paveiq.config import ARTIFACTS_DIR
from paveiq.dashboard import data, mapview, whatif
from paveiq.models.registry import load_scorer

st.set_page_config(page_title="PaveIQ", page_icon="\U0001f6b6", layout="wide")


def _find_latest_artifact() -> Path:
    """Return the newest ``*_scorer*.json`` in ``artifacts/``."""
    candidates = sorted(ARTIFACTS_DIR.glob("*_scorer*.json"))
    if not candidates:
        raise FileNotFoundError(f"no scorer artifact in {ARTIFACTS_DIR}; run `python -m paveiq.models.train` first")
    return max(candidates, key=lambda p: p.stat().st_mtime)


@st.cache_data
def _load_segments():
    return data.load_scored_segments()


@st.cache_data
def _load_wards():
    return data.load_ward_polygons()


@st.cache_resource
def _load_scorer():
    return load_scorer(_find_latest_artifact())


SURFACE_UPGRADE_OPTIONS = {"Compacted (0.6)": 0.6, "Paved (1.0)": 1.0}
WHOLE_WARD_LABEL = "Whole ward (bulk-simulate)"


def render_map_tab(scored, wards, ward_summary_df):
    col_controls, col_map = st.columns([1, 3])
    with col_controls:
        max_elevation = st.slider(
            "Height exaggeration (max elevation, m)",
            min_value=20,
            max_value=300,
            value=int(mapview.MAX_ELEVATION_M),
            step=10,
        )
        show_wards = st.checkbox("Overlay ward choropleth", value=False)
        st.caption("Height = worse score → taller. Color: red (poor) → green (good).")

    layers = [mapview.build_segment_layer(scored, max_elevation_m=max_elevation)]
    if show_wards:
        wards_with_scores = wards.merge(
            ward_summary_df[["ward_id", "mean_score"]], on="ward_id", how="left"
        )
        layers.append(mapview.build_ward_choropleth_layer(wards_with_scores))

    with col_map:
        st.pydeck_chart(
            mapview.make_deck(
                layers,
                tooltip={"html": "<b>{highway}</b><br/>Score: {score}"},
            )
        )


def render_leaderboard_tab(scored, ward_summary_df):
    st.subheader("Ward leaderboard (worst first)")
    st.dataframe(ward_summary_df, width="stretch")

    st.subheader("Worst individual segments")
    n = st.slider("Number of segments to show", 5, 100, 20, step=5)
    st.dataframe(data.leaderboard(scored, n=n), width="stretch")


def render_whatif_tab(scored, ward_summary_df, scorer):
    ward_options = ward_summary_df[["ward_id", "ward_name"]].drop_duplicates()
    ward_labels = {row.ward_id: f"{row.ward_name} ({row.ward_id})" for row in ward_options.itertuples()}
    selected_ward_id = st.selectbox(
        "Ward", options=list(ward_labels), format_func=lambda wid: ward_labels[wid]
    )

    segments_in_ward = scored.loc[scored["ward_id"] == selected_ward_id]
    segment_choice = st.selectbox(
        "Segment (optional — leave as whole ward to bulk-simulate)",
        options=[WHOLE_WARD_LABEL] + list(segments_in_ward["osmid"]),
    )
    if segment_choice == WHOLE_WARD_LABEL:
        scope, target = "ward", selected_ward_id
    else:
        scope, target = "segment", segment_choice

    st.markdown("**Hypothetical changes**")
    c1, c2 = st.columns(2)
    with c1:
        apply_surface = st.checkbox("Upgrade surface")
        surface_label = st.selectbox(
            "Target surface", options=list(SURFACE_UPGRADE_OPTIONS), disabled=not apply_surface
        )
        apply_sidewalk = st.checkbox("Add sidewalk")
    with c2:
        apply_widen = st.checkbox("Widen path")
        width_target = st.slider("Target width (m)", 0.5, 3.0, 1.5, step=0.1, disabled=not apply_widen)
        apply_pedestrianize = st.checkbox("Pedestrianize (convert to dedicated footway)")

    toggles = {}
    if apply_surface:
        toggles["surface_upgrade"] = SURFACE_UPGRADE_OPTIONS[surface_label]
    if apply_sidewalk:
        toggles["add_sidewalk"] = "explicit_present"
    if apply_widen:
        toggles["widen_path"] = width_target
    if apply_pedestrianize:
        toggles["pedestrianize"] = 1.0

    if not toggles:
        st.info("Toggle a hypothetical change above to see the recomputed score.")
        return

    original = whatif.select_scope(scored, scope, target)
    modified, affected = whatif.apply_hypothetical(scored, toggles, scope, target)
    before_mean = scorer.predict(original).mean()
    after_mean = scorer.predict(modified).mean()

    st.metric(
        "Score" if scope == "segment" else "Mean ward score",
        f"{after_mean:.1f}",
        delta=f"{after_mean - before_mean:+.1f}",
    )
    for toggle_name, n_affected in affected.items():
        st.caption(f"{toggle_name}: {n_affected} of {len(original)} segments affected")


def main():
    st.title("PaveIQ — Footpath Health Dashboard")

    try:
        scored = _load_segments()
        wards = _load_wards()
        scorer = _load_scorer()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
        return

    ward_summary_df = data.ward_summary(scored)

    st.caption(
        f"{len(scored):,} segments · city-wide mean score {scored['score'].mean():.1f} "
        f"· {ward_summary_df['ward_id'].nunique()} wards"
    )

    tab_map, tab_leaderboard, tab_whatif = st.tabs(["Map", "Ward Leaderboard", "What-if"])
    with tab_map:
        render_map_tab(scored, wards, ward_summary_df)
    with tab_leaderboard:
        render_leaderboard_tab(scored, ward_summary_df)
    with tab_whatif:
        render_whatif_tab(scored, ward_summary_df, scorer)


if __name__ == "__main__":
    main()

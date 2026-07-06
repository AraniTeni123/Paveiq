"""PaveIQ Streamlit dashboard.

Thin UI glue only — every function with real logic lives in
``data.py``/``mapview.py``/``whatif.py`` and is unit-tested there.
``theme.py`` supplies the dark palette and HTML/CSS card builders. This
module wires them to Streamlit widgets: a sidebar for navigation and
always-visible key stats, and a main area for whichever view is active
(3D map, ward leaderboard, what-if simulator).

Run with::

    streamlit run src/paveiq/dashboard/app.py
"""

from __future__ import annotations

import pydeck as pdk
import streamlit as st

from paveiq.dashboard import data, mapview, theme, whatif
from paveiq.models.registry import load_scorer

st.set_page_config(page_title="PaveIQ", page_icon="\U0001f6b6", layout="wide")

VIEWS = ("Map", "Ward Leaderboard", "What-if")


@st.cache_data
def _load_segments():
    return data.load_scored_segments()


@st.cache_data
def _load_wards():
    return data.load_ward_polygons()


@st.cache_resource
def _load_scorer():
    return load_scorer(data._find_latest_artifact())


SURFACE_UPGRADE_OPTIONS = {"Compacted (0.6)": 0.6, "Paved (1.0)": 1.0}
WHOLE_WARD_LABEL = "Whole ward (bulk-simulate)"


def _score_status(score: float) -> str:
    """"good"/"bad" card coloring for a mean score, split at the scale midpoint."""
    return "good" if score >= theme.GOOD_SCORE_THRESHOLD else "bad"


def render_sidebar(scored, ward_summary_df) -> str:
    """Sidebar: wordmark, nav, and the three always-visible key stats.

    Returns the selected view name.
    """
    with st.sidebar:
        st.markdown('<div class="pq-wordmark">\U0001f6b6 PaveIQ</div>', unsafe_allow_html=True)
        view = st.radio("Navigate", VIEWS, label_visibility="collapsed")

        st.markdown(
            theme.metric_card_html("Segments", f"{len(scored):,}", status="neutral"),
            unsafe_allow_html=True,
        )
        mean_score = scored["score"].mean()
        st.markdown(
            theme.metric_card_html(
                "Mean score", f"{mean_score:.1f}", status=_score_status(mean_score)
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            theme.metric_card_html(
                "Wards", f"{ward_summary_df['ward_id'].nunique()}", status="neutral"
            ),
            unsafe_allow_html=True,
        )
    return view


def render_header_cards(scored, ward_summary_df) -> None:
    """The 4-card horizontal row: segments, mean score, wards, worst ward."""
    worst_ward = ward_summary_df.iloc[0]
    mean_score = scored["score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            theme.metric_card_html("Segments", f"{len(scored):,}", status="neutral"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            theme.metric_card_html(
                "Mean score", f"{mean_score:.1f}", status=_score_status(mean_score)
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            theme.metric_card_html(
                "Wards", f"{ward_summary_df['ward_id'].nunique()}", status="neutral"
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            theme.metric_card_html(
                "Worst ward",
                f"{worst_ward['mean_score']:.1f}",
                status="bad",
                sublabel=str(worst_ward["ward_name"]),
            ),
            unsafe_allow_html=True,
        )


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
                map_style=pdk.map_styles.CARTO_DARK,
                tooltip={"html": "<b>{highway}</b><br/>Score: {score}"},
            )
        )


def render_leaderboard_tab(scored, ward_summary_df):
    st.subheader("Ward leaderboard (worst first)")
    st.dataframe(
        ward_summary_df.style.background_gradient(
            cmap=theme.SCORE_GRADIENT_CMAP, subset=["mean_score"], vmin=0, vmax=100
        ).format({"mean_score": "{:.1f}", "pct_poor": "{:.1f}", "total_length_m": "{:.0f}"}),
        width="stretch",
    )

    st.subheader("Worst individual segments")
    n = st.slider("Number of segments to show", 5, 100, 20, step=5)
    worst_segments = data.leaderboard(scored, n=n)
    st.dataframe(
        worst_segments.style.background_gradient(
            cmap=theme.SCORE_GRADIENT_CMAP, subset=["score"], vmin=0, vmax=100
        ).format({"score": "{:.1f}", "length_m": "{:.0f}"}),
        width="stretch",
    )


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

    label = "Score" if scope == "segment" else "Mean ward score"
    st.markdown(
        theme.before_after_card_html(label, before_mean, after_mean),
        unsafe_allow_html=True,
    )
    st.write("")
    for toggle_name, n_affected in affected.items():
        st.caption(f"{toggle_name}: {n_affected} of {len(original)} segments affected")


def main():
    st.markdown(theme.inject_global_css(), unsafe_allow_html=True)

    try:
        scored = _load_segments()
        wards = _load_wards()
        scorer = _load_scorer()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
        return

    ward_summary_df = data.ward_summary(scored)
    view = render_sidebar(scored, ward_summary_df)

    st.title("PaveIQ — Footpath Health Dashboard")
    render_header_cards(scored, ward_summary_df)
    st.write("")

    if view == "Map":
        render_map_tab(scored, wards, ward_summary_df)
    elif view == "Ward Leaderboard":
        render_leaderboard_tab(scored, ward_summary_df)
    else:
        render_whatif_tab(scored, ward_summary_df, scorer)


if __name__ == "__main__":
    main()

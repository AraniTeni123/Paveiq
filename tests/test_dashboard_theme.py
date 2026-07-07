"""Offline tests for ``paveiq.dashboard.theme``.

Pure string-content checks — theme.py builds HTML/CSS strings and never
imports streamlit, so there's no DOM to render; these assert the
generated markup contains what it's supposed to.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from paveiq.dashboard import theme


# --- metric_card_html ----------------------------------------------------


def test_metric_card_html_includes_label_value_sublabel():
    html = theme.metric_card_html("Segments", "21,947", status="neutral", sublabel="Koramangala")
    assert "Segments" in html
    assert "21,947" in html
    assert "Koramangala" in html


def test_metric_card_html_omits_sublabel_when_not_given():
    html = theme.metric_card_html("Wards", "68")
    assert "pq-sublabel" not in html


def test_metric_card_html_good_status_uses_good_class_and_color_defined():
    html = theme.metric_card_html("Mean score", "63.7", status="good")
    assert "pq-good" in html
    assert theme.GOOD in theme.inject_global_css()  # color is wired into the CSS


def test_metric_card_html_bad_status_uses_bad_class():
    html = theme.metric_card_html("Mean score", "31.0", status="bad")
    assert "pq-bad" in html


def test_metric_card_html_neutral_status_uses_neutral_class():
    html = theme.metric_card_html("Segments", "21,947", status="neutral")
    assert "pq-neutral" in html


def test_metric_card_html_rejects_invalid_status():
    with pytest.raises(ValueError, match="status must be"):
        theme.metric_card_html("X", "1", status="ok")


# --- before_after_card_html ------------------------------------------------


def test_before_after_card_shows_both_values():
    html = theme.before_after_card_html("Score", 45.1, 63.7)
    assert "45.1" in html
    assert "63.7" in html
    assert "+18.6" in html


def test_before_after_card_improvement_is_up_and_good_colored():
    html = theme.before_after_card_html("Score", 40.0, 60.0)
    assert "pq-up" in html
    assert "▲" in html
    assert "pq-down" not in html


def test_before_after_card_regression_is_down_and_bad_colored():
    html = theme.before_after_card_html("Score", 60.0, 40.0)
    assert "pq-down" in html
    assert "▼" in html
    assert "-20.0" in html


def test_before_after_card_no_change_boundary_case():
    html = theme.before_after_card_html("Score", 50.0, 50.0)
    assert "+0.0" in html
    # Boundary (delta == 0) renders as the "up"/neutral treatment, not "down".
    assert "pq-down" not in html


# --- inject_global_css ---------------------------------------------------


def test_inject_global_css_includes_font_import():
    css = theme.inject_global_css()
    assert "JetBrains+Mono" in css


def test_inject_global_css_includes_display_font_import_and_header_rule():
    css = theme.inject_global_css()
    assert "Space+Grotesk" in css
    assert theme.FONT_DISPLAY in css
    # Headers and mono numbers must be different font stacks -- that's the point.
    assert theme.FONT_DISPLAY != theme.FONT_MONO


def test_inject_global_css_includes_palette_hex_values():
    css = theme.inject_global_css()
    for hex_color in (theme.BG, theme.SURFACE, theme.ACCENT, theme.GOOD, theme.BAD):
        assert hex_color in css


# --- map_legend_html / insight_card_html -----------------------------------


def test_map_legend_html_explains_height_and_color_encoding():
    html = theme.map_legend_html()
    assert "pq-legend" in html
    assert "Poor" in html and "Good" in html
    assert "Height" in html


def test_legend_swatch_css_gradient_uses_score_gradient_colors():
    """The swatch's gradient (defined in the CSS rule, not the HTML snippet
    itself) should be sampled from SCORE_GRADIENT_CMAP, not a hardcoded
    placeholder -- spot-check the reddest sampled stop appears."""
    css = theme.inject_global_css()
    reddest = "#%02x%02x%02x" % tuple(int(c * 255) for c in theme.SCORE_GRADIENT_CMAP(0.0)[:3])
    assert reddest in css
    assert "pq-legend-swatch" in css


def test_insight_card_html_wraps_text_with_tag():
    html = theme.insight_card_html("HAL Airport sits 11.8 points below the mean.")
    assert "pq-insight" in html
    assert "HAL Airport sits 11.8 points below the mean." in html
    assert "Insight" in html


# --- SCORE_GRADIENT_CMAP ---------------------------------------------------


def test_score_gradient_cmap_is_usable_by_pandas_styler():
    import pandas as pd

    df = pd.DataFrame({"score": [10.0, 50.0, 90.0]})
    styled = df.style.background_gradient(cmap=theme.SCORE_GRADIENT_CMAP, subset=["score"], vmin=0, vmax=100)
    html = styled.to_html()
    assert "background-color" in html


# --- .streamlit/config.toml ------------------------------------------------


def test_streamlit_config_toml_is_valid_with_expected_keys():
    config_path = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    theme_section = data["theme"]
    assert theme_section["base"] == "dark"
    assert theme_section["backgroundColor"] == theme.BG
    assert theme_section["secondaryBackgroundColor"] == theme.SURFACE
    assert theme_section["primaryColor"] == theme.ACCENT

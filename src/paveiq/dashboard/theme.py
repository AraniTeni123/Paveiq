"""Dark-theme palette and HTML/CSS builders for the dashboard.

Pure string templating — no ``streamlit`` import — so it's unit-testable
without a browser, same convention as ``data.py``/``mapview.py``/``whatif.py``.
``app.py`` is the only module that actually calls
``st.markdown(..., unsafe_allow_html=True)``.

The palette was not eyeballed: every color here was checked with the
dataviz skill's WCAG-contrast / OKLCH validator against the real
``BG``/``SURFACE`` this app renders on (not a generic default surface).

    node validate_palette.js "#00d9ff,#0ca30c,#e66767" --mode dark \\
        --surface "#0e1117" --pairs all
    # -> CVD separation PASS (worst adjacent ΔE 12.4), contrast PASS (all >=3:1)

Plain WCAG text-contrast ratios (``contrast(a, b)`` from that same script)
against ``BG`` / ``SURFACE``:

=====================  =======  =========  ===========
role                   hex      on BG      on SURFACE
=====================  =======  =========  ===========
primary text           #ffffff  18.90      17.30
muted text             #9aa4b2   7.49         --
accent                 #00d9ff  11.13      10.19
good status            #0ca30c   5.63       5.16
bad status             #e66767   5.85       5.35
=====================  =======  =========  ===========

All clear WCAG AA (4.5:1 text / 3:1 UI component). ``ACCENT`` alone fails
the validator's *categorical* lightness band (OKLCH L=0.816 — too light
for a data-encoding series) — irrelevant here since it's a UI/brand
accent, not a data series; the skill's checks explicitly scope lone
UI/status colors out of that particular check.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

# --- Palette -----------------------------------------------------------

BG = "#0e1117"
SURFACE = "#161b22"
TEXT = "#ffffff"
TEXT_MUTED = "#9aa4b2"
ACCENT = "#00d9ff"
ACCENT_GLOW = "#14ffec"
GOOD = "#0ca30c"
BAD = "#e66767"

# score >= this -> "good" card coloring, else "bad". 50 is the scale
# midpoint (SCORE_MIN/MAX are 0/100) and matches the real dataset's
# observed median (50.0) -- see models/heuristic.py.
GOOD_SCORE_THRESHOLD = 50

FONT_MONO = "'JetBrains Mono', monospace"
FONT_DISPLAY = "'Space Grotesk', sans-serif"


def _blend(hex_color: str, bg: str, alpha: float) -> str:
    """Blend ``hex_color`` toward ``bg`` by ``alpha`` (1.0 = no blend)."""

    def _to_rgb(h: str) -> tuple:
        h = h.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    c, b = _to_rgb(hex_color), _to_rgb(bg)
    blended = tuple(round(alpha * c[i] + (1 - alpha) * b[i]) for i in range(3))
    return "#%02x%02x%02x" % blended


def _build_score_gradient_cmap() -> LinearSegmentedColormap:
    """Blend matplotlib's ``RdYlGn`` 40% toward ``BG``.

    Raw ``RdYlGn`` (what the pydeck map uses) gets too light at the
    yellow midpoint for white text on a dark background (contrast
    collapses to ~1.0-1.6 there). Blending 40% into ``BG`` keeps the
    same red->yellow->green *direction* — visual consistency with the
    map — while guaranteeing white-text contrast >= 5.09:1 at every
    stop (the worst case, at the yellow midpoint).
    """
    base = colormaps["RdYlGn"]
    stops = np.linspace(0, 1, 9)
    blended = [_blend("#%02x%02x%02x" % tuple(int(c * 255) for c in base(s)[:3]), BG, 0.4) for s in stops]
    return LinearSegmentedColormap.from_list("paveiq_score_gradient", blended)


SCORE_GRADIENT_CMAP = _build_score_gradient_cmap()


def _gradient_css_stops(n: int = 5) -> str:
    """Sample ``SCORE_GRADIENT_CMAP`` into an ``a, b, c, ...`` CSS gradient stop list."""
    hexes = [
        "#%02x%02x%02x" % tuple(int(c * 255) for c in SCORE_GRADIENT_CMAP(s)[:3])
        for s in np.linspace(0, 1, n)
    ]
    return ", ".join(hexes)


# --- CSS -----------------------------------------------------------------


def inject_global_css() -> str:
    """One ``<style>`` block: font import, palette variables, widget overrides."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
    --pq-bg: {BG};
    --pq-surface: {SURFACE};
    --pq-text: {TEXT};
    --pq-text-muted: {TEXT_MUTED};
    --pq-accent: {ACCENT};
    --pq-accent-glow: {ACCENT_GLOW};
    --pq-good: {GOOD};
    --pq-bad: {BAD};
}}

.stApp {{
    background-color: var(--pq-bg);
}}

[data-testid="stSidebar"] {{
    background-color: var(--pq-surface);
    border-right: 1px solid rgba(255,255,255,0.08);
}}

[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.95rem;
}}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
    color: var(--pq-accent);
}}

.stApp h1, .stApp h2, .stApp h3, .pq-wordmark {{
    font-family: {FONT_DISPLAY};
}}

.pq-wordmark {{
    font-weight: 700;
    font-size: 1.4rem;
    color: var(--pq-accent);
    letter-spacing: 0.02em;
    padding: 4px 0 12px 0;
}}

.pq-card {{
    background: var(--pq-surface);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid var(--pq-accent);
    padding: 14px 16px;
    margin-bottom: 10px;
}}

.pq-card.pq-good {{
    border-left-color: var(--pq-good);
    box-shadow: 0 0 12px rgba(12,163,12,0.15);
}}

.pq-card.pq-bad {{
    border-left-color: var(--pq-bad);
    box-shadow: 0 0 12px rgba(230,103,103,0.18);
}}

.pq-card.pq-neutral {{
    border-left-color: var(--pq-accent);
    box-shadow: 0 0 12px rgba(0,217,255,0.12);
}}

.pq-card .pq-label {{
    color: var(--pq-text-muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}}

.pq-card .pq-value {{
    font-family: {FONT_MONO};
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--pq-text);
    line-height: 1.2;
}}

.pq-card .pq-sublabel {{
    color: var(--pq-text-muted);
    font-size: 0.8rem;
    margin-top: 2px;
}}

.pq-before-after {{
    background: var(--pq-surface);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 24px 28px;
    text-align: center;
}}

.pq-before-after .pq-label {{
    color: var(--pq-text-muted);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 10px;
}}

.pq-before-after .pq-numbers {{
    font-family: {FONT_MONO};
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--pq-text-muted);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
}}

.pq-before-after .pq-after {{
    color: var(--pq-text);
}}

.pq-before-after .pq-arrow {{
    font-size: 1.6rem;
}}

.pq-before-after .pq-arrow.pq-up {{
    color: var(--pq-good);
}}

.pq-before-after .pq-arrow.pq-down {{
    color: var(--pq-bad);
}}

.pq-before-after .pq-delta {{
    margin-top: 8px;
    font-family: {FONT_MONO};
    font-size: 1.1rem;
}}

.pq-before-after .pq-delta.pq-up {{
    color: var(--pq-good);
}}

.pq-before-after .pq-delta.pq-down {{
    color: var(--pq-bad);
}}

.pq-legend {{
    background: var(--pq-surface);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 10px 16px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    color: var(--pq-text-muted);
    font-size: 0.85rem;
}}

.pq-legend .pq-legend-swatch {{
    width: 120px;
    height: 10px;
    border-radius: 5px;
    background: linear-gradient(to right, {_gradient_css_stops()});
}}

.pq-insight {{
    background: var(--pq-surface);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid var(--pq-accent-glow);
    padding: 12px 16px;
    margin-bottom: 14px;
    color: var(--pq-text);
    font-size: 0.95rem;
    line-height: 1.5;
}}

.pq-insight .pq-insight-tag {{
    color: var(--pq-accent-glow);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
    display: block;
}}
</style>
"""


# --- HTML component builders -----------------------------------------------


def metric_card_html(label: str, value: str, status: str = "neutral", sublabel: str | None = None) -> str:
    """One metric card's HTML: colored left border + glow per ``status``.

    ``status`` is ``"good"``, ``"bad"``, or ``"neutral"`` (default) — see
    the module docstring / plan for which metrics get which.
    """
    if status not in ("good", "bad", "neutral"):
        raise ValueError(f"status must be 'good', 'bad', or 'neutral', got {status!r}")
    sublabel_html = f'<div class="pq-sublabel">{sublabel}</div>' if sublabel else ""
    return (
        f'<div class="pq-card pq-{status}">'
        f'<div class="pq-label">{label}</div>'
        f'<div class="pq-value">{value}</div>'
        f"{sublabel_html}"
        f"</div>"
    )


def map_legend_html() -> str:
    """A legend card explaining the map's height/color encoding.

    Rendered directly above the pydeck chart (not a floating overlay on
    the canvas — robust, no CSS-over-iframe fragility, and still can't
    be missed the way a small gray caption could be).
    """
    return (
        '<div class="pq-legend">'
        '<div class="pq-legend-swatch"></div>'
        "<span>Poor &rarr; Good</span>"
        "<span>&middot;</span>"
        "<span>Height = worse score, taller</span>"
        "</div>"
    )


def insight_card_html(text: str) -> str:
    """A callout card for an auto-generated interpretive sentence.

    Visually distinct from ``metric_card_html`` (accent-glow left border,
    no big mono number) so it reads as commentary, not another data point.
    """
    return f'<div class="pq-insight"><span class="pq-insight-tag">Insight</span>{text}</div>'


def before_after_card_html(label: str, before: float, after: float) -> str:
    """A large before -> after comparison card with a colored delta arrow."""
    delta = after - before
    direction = "up" if delta > 0 else "down" if delta < 0 else "up"
    arrow = "▲" if delta >= 0 else "▼"
    sign = "+" if delta >= 0 else ""
    return (
        f'<div class="pq-before-after">'
        f'<div class="pq-label">{label}</div>'
        f'<div class="pq-numbers">'
        f"<span>{before:.1f}</span>"
        f'<span class="pq-arrow pq-{direction}">{arrow}</span>'
        f'<span class="pq-after">{after:.1f}</span>'
        f"</div>"
        f'<div class="pq-delta pq-{direction}">{sign}{delta:.1f}</div>'
        f"</div>"
    )

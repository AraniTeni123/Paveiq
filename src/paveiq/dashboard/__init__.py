"""Streamlit dashboard over the scored footpath network.

Reads the scored Parquet from ``scoring.score_city`` and the ward
polygons from ``data_ingestion.ward_boundaries``, and presents a 3D
map, a ward-level leaderboard, and a what-if panel for simulating
hypothetical feature changes against the live scorer.

Only ``app.py`` imports ``streamlit`` — ``data.py``, ``mapview.py``,
and ``whatif.py`` are plain pandas/geopandas/pydeck and are covered
by ordinary pytest, no browser required.
"""

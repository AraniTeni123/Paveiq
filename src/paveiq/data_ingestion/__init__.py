"""Data ingestion: fetch and normalize raw geospatial sources.

Stage 1 of the pipeline. Each submodule targets one source
(OpenStreetMap, BBMP wards, raster layers, citizen reports, etc.)
and writes a normalized file to ``data/raw/<source>/``.
"""

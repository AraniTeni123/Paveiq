"""Feature engineering over footpath segments.

Stage 2 of the pipeline. Reads normalized raw data and produces
a per-segment feature table (Parquet / GeoPackage) consumed by
the modeling stage.
"""

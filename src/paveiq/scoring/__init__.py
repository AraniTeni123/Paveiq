"""Apply the trained model to the full city network.

Stage 4 of the pipeline. Reads the trained model artifact and
the full (unlabeled) feature table, and writes a scored
geospatial layer to ``data/processed/``.
"""

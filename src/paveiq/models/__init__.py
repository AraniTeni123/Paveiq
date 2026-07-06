"""Model training, evaluation, and persistence.

Stage 3 of the pipeline. Reads the feature table, fits a model
against labeled footpath-condition data, and serializes the
trained model to ``artifacts/``.
"""

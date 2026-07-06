"""Offline tests for ``paveiq.models.registry``."""

from __future__ import annotations

import json

import pytest

from paveiq.models.heuristic import HeuristicScorer
from paveiq.models.registry import load_scorer


def test_load_scorer_dispatches_to_heuristic(tmp_path):
    path = tmp_path / "scorer.json"
    HeuristicScorer().save(path)
    scorer = load_scorer(path)
    assert isinstance(scorer, HeuristicScorer)


def test_load_scorer_raises_on_unknown_model_type(tmp_path):
    path = tmp_path / "scorer.json"
    path.write_text(json.dumps({"model_type": "not_a_real_model"}))
    with pytest.raises(ValueError, match="unknown model_type"):
        load_scorer(path)

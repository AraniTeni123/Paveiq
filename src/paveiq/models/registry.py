"""Scorer interface and artifact-loading dispatch.

The rest of the pipeline (``scoring/score_city.py``, the dashboard)
never imports a concrete scorer class directly. It calls
``load_scorer(path).predict(df)``. This is the swap-point: today
``path`` points at a :class:`~paveiq.models.heuristic.HeuristicScorer`
JSON artifact; once labeled data exists and a real sklearn/xgboost
model is trained, a new artifact with a different ``model_type``
can be dropped in and nothing outside ``models/`` has to change.

Adding a new scorer type
------------------------
1. Implement a class with a ``predict(df) -> np.ndarray`` method and
   a ``save`` / classmethod ``load`` pair (see ``HeuristicScorer``).
2. Register its loader: ``SCORER_LOADERS["my_model_v1"] = MyScorer.load``.
3. Artifacts written by that class must include a top-level
   ``"model_type": "my_model_v1"`` key so ``load_scorer`` can dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd


class Scorer(Protocol):
    """Anything that turns a segment feature table into 0-100 scores."""

    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


def _load_heuristic(path: Path):
    from paveiq.models.heuristic import HeuristicScorer

    return HeuristicScorer.load(path)


# model_type (as stored in the artifact JSON) -> loader.
SCORER_LOADERS: dict[str, Callable[[Path], Scorer]] = {
    "heuristic_v1": _load_heuristic,
}


def load_scorer(path: Path) -> Scorer:
    """Read a scorer artifact and dispatch to the right loader by ``model_type``."""
    path = Path(path)
    model_type = json.loads(path.read_text()).get("model_type")
    if model_type not in SCORER_LOADERS:
        raise ValueError(
            f"unknown model_type {model_type!r} in {path}; "
            f"registered types are {sorted(SCORER_LOADERS)}"
        )
    return SCORER_LOADERS[model_type](path)

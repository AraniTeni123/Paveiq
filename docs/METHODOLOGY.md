# PaveIQ Methodology

This document explains the *why* behind PaveIQ's design decisions — reconstructed
from the git history and codebase for a reader who's technical but new to the
project. It isn't a design doc written up front; it's a retrospective account of
what was tried, what broke, and what that revealed.

## 1. Why OSMnx's features API, not the graph API

OSMnx offers two fundamentally different ways to pull data from OpenStreetMap:

- **`graph_from_place`/`graph_from_polygon`** — builds a routable street-network
  *graph* (nodes and directed edges), designed for routing and connectivity
  analysis. It's scoped to `highway=*`-tagged ways because that's what a road
  network is made of, and it consolidates/simplifies intersections along the way.
- **`features_from_polygon`** (used here, in `data_ingestion/osm_loader.py`) —
  a general-purpose OSM element extractor driven by an arbitrary tag filter. It
  has no notion of routing topology; it just returns everything matching the
  filter.

PaveIQ needs the second one, for a reason specific to how Bengaluru's sidewalks
are actually tagged: a sidewalk is frequently recorded as a *tag on its parent
road* (`sidewalk=both` on a `highway=residential` way), not as its own
`highway=footway` way. `osm_loader._tags_of_interest()` builds an **inclusive OR
filter**:

```python
{
    "highway": list(FOOTPATH_HIGHWAY_VALUES),  # footway..trunk, 15 values
    "sidewalk": True,   # "this key exists, any value"
    "surface": True,
    "width": True,
    "smoothness": True,
}
```

This says "give me a way if it's one of these highway types, **or** it carries
a sidewalk/surface/width/smoothness tag at all" — capturing sidewalks-on-roads
that a highway-only filter would miss entirely. The graph API has no equivalent
of this: it's built around one tag key (`highway`) as the organizing structure
of the network, not an arbitrary multi-tag OR query. The features API's
flexibility is the whole reason it was chosen — PaveIQ cares about *attributes
attached to ways*, not routing topology.

## 2. OSM tag coverage in Bengaluru: sparser than routing data usually is

The very first commit's docstring already predicted this ("Bengaluru's OSM
coverage of these tags is known to be sparse (<5% of ways carry a `sidewalk`
tag)"), and the first real pull confirmed it. Against the actual Koramangala-area
data (`data/raw/12.9300_77.5500_12.9900_77.6500_footpaths.geojson`, 21,947 ways):

| tag | non-null | coverage |
|---|---|---|
| `sidewalk` | 807 | 3.7% |
| `surface` | 4,745 | 21.6% |
| `width` | 86 | 0.4% |
| `smoothness` | 1,338 | 6.1% |
| **any of the four** | 4,943 | **22.5%** |

Read that last row carefully: only 22.5% of ways carry *any* of the four
quality-relevant tags at all — meaning **over three-quarters of the network has
zero direct signal** about surface, sidewalk presence, width, or smoothness.
This single number is the load-bearing fact behind almost every downstream
design decision in this project:

- It's why `sidewalk_presence` (§3) is a 5-bucket *inference*, not a raw tag
  passthrough — with 96.3% of ways missing an explicit `sidewalk` tag, the
  feature has to fall back to a highway-type prior or it would be useless.
- It's why `width_m` was deprioritized to the lowest scoring weight (§3, §4) —
  0.4% coverage means it's nearly always missing.
- It's why the scoring model is a heuristic, not a trained model (§4) — sparse
  *feature* coverage is a separate problem from missing *labels*, but both
  point the same direction: there isn't enough confirmed ground truth yet to
  fit or evaluate a model with confidence.
- It's why `n_features_observed` exists as a visible confidence signal in the
  scored output — a score built from 2 observed features and 2 neutral guesses
  should visibly read differently from one built from all 4.

## 3. Feature engineering choices

All of this lives in `src/paveiq/features/build_features.py`.

### `highway_likelihood` — an ordinal prior, not a boolean

Maps the `highway` tag to a 0–1 "is this actually a place a footpath would be"
score, because `highway=primary` and `highway=footway` are not just different —
they're opposite ends of "how likely is a pedestrian facility here":

```
footway        1.00      cycleway       0.60      primary        0.10
path           0.95      residential    0.50      trunk          0.05
pedestrian     0.90      service        0.30
living_street  0.85      unclassified   0.30
steps          0.85      tertiary       0.20
                          construction   0.20
```

`*_link` variants (e.g. `primary_link`) score 0.05 below their parent class —
a slip road is marginally more pedestrian-relevant than the road it serves,
but not by much. Unknown/null `highway` values return `NaN`, not `0.0` — this
distinction matters: `0.0` would be a confident "definitely not a footpath"
signal, while `NaN` means "no signal at all," and the scoring model (§4) treats
those very differently (a missing value gets a neutral 0.5, not a penalty).

### `surface_quality` — a deliberately coarse 3-tier ordinal

OSM's `surface` tag has dozens of real-world values (`asphalt`, `paving_stones`,
`sett`, `cobblestone`, `compacted`, `unpaved`, `dirt`, `gravel`, ...). Rather
than trying to rank all of them, they're collapsed to three tiers by "can a
wheelchair or stroller roll on this":

- **1.0 (paved)** — `asphalt`, `paving_stones`, `concrete`, `paved`, `sett`,
  `cobblestone`, `tiles`, `clinker_plates`, `stone`, `cement`, `concrete:plates`
- **0.6 (compacted)** — `compacted`
- **0.2 (unpaved)** — `unpaved`, `dirt`, `ground`, `gravel`, `rock`, `mud`
- **NaN** — anything else, or the tag is missing (21.6% coverage, see §2)

Three tiers instead of a finer scale because OSM's own tagging doesn't reliably
distinguish finer gradations in practice — encoding false precision the data
doesn't support would just be noise.

### `sidewalk_presence` — inference, because the tag is almost always absent

With `sidewalk` present on only 3.7% of ways, a raw passthrough of the tag would
leave 96.3% of segments with no signal. `sidewalk_presence` instead folds in a
highway-type prior, producing 5 mutually exclusive buckets:

1. **`explicit_present`** — `sidewalk` tag says `both`/`left`/`right`/`separate` (or any other non-"no" value)
2. **`explicit_absent`** — `sidewalk=no`
3. **`implicit_present`** — no sidewalk tag, but `highway` is `footway`/`path`/`pedestrian`/`steps`/`cycleway` — the way *is* the footpath, so nobody tags a sidewalk on a sidewalk
4. **`likely_present`** — no sidewalk tag, `highway` is `living_street`/`residential`/`service`/`unclassified` — a prior, not a measurement
5. **`unlikely`** — everything else (arterial roads, unknown highway) without an explicit tag

This is the one feature with 100% coverage by construction (every row lands in
exactly one bucket), which is part of why it carries the second-highest scoring
weight (§4) — it's the most reliable signal available, even though most of that
reliability comes from inference rather than direct observation.

### `width_m` — parsed carefully, then deprioritized anyway

OSM's `width` tag is free-text and genuinely dirty in the wild — `"2"`, `"2.5"`,
`"100 cm"`, `"25 ft"`, `"20'"`, `"24\""`, `"500 mm"` all appear. `parse_width_m`
handles all of these via a regex + unit-conversion table, clamping to a
plausible physical range (0.1–100 m) to reject data-entry errors. Despite that
care, `width_m` ends up the *least*-weighted feature in the scoring model
(§4) for one reason: **0.4% coverage** (86 of 21,947 ways). No amount of
parsing sophistication fixes a feature that's almost never present — the
engineering effort here is about correctness for the rare row that *does* have
it, not about it carrying the score.

## 4. Why a transparent heuristic, not a trained model

`src/paveiq/models/heuristic.py`'s docstring states the constraint plainly:
Stage 3 needs "labeled footpath-condition data" to fit a real
regression/classifier, and **none exists in the repo** — no citizen reports, no
field verification, no score column anywhere. That's a different gap from the
tag sparsity in §2 (missing *features* vs. missing *labels*), but both point
the same way: there isn't yet a sound basis for training or evaluating a model.

`HeuristicScorer` is the interim answer: a hand-tuned weighted sum over the
four engineered features, transparent enough to sanity-check by eye —

```
score = 100 × (0.35·surface + 0.30·sidewalk + 0.15·width + 0.20·highway)
```

— with missing sub-scores **neutral-imputed to 0.5** rather than
excluded-and-renormalized. That choice matters: renormalizing weights per row
would make two segments with identical *observed* values score differently
depending on which fields happened to be missing, which is hard to explain to
the advocacy audience this tool serves. The practical cost, given `width_m`'s
0.4% coverage, is that its 0.15 weight is an almost-constant offset for nearly
every segment — a documented limitation, not a hidden one (`n_features_observed`,
0–4, ships alongside every score specifically so this is visible rather than
silently assumed away).

**The interface is the actual point.** `models/registry.py` defines the
contract the rest of the pipeline depends on:

```python
class Scorer(Protocol):
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...
```

`scoring/score_city.py` and the dashboard only ever call
`registry.load_scorer(artifact_path).predict(df)` — never `HeuristicScorer`
directly. Artifacts are plain JSON (git-diffable, human-auditable — fits
"transparent scorer," and avoids sklearn pickle-version-compat risk) with a
`model_type` field the registry dispatches on. When real labels exist, a future
`SklearnScorer` implementing the same `predict(df) -> np.ndarray` shape,
serialized via joblib, registers itself as `SCORER_LOADERS["sklearn_gbm_v1"]`
and **nothing in `scoring/` or `dashboard/` changes**. The heuristic isn't a
placeholder bolted on before the "real" architecture — the registry/Protocol
split *is* the architecture, designed so the model can change without the
system around it noticing.

## 5. Ward-boundary join methodology

`src/paveiq/data_ingestion/ward_boundaries.py` joins each footpath segment to
its BBMP ward, sourced from DataMeet's `Municipal_Spatial_Data` (2022 KGIS
delimitation, 243 wards, CC-BY-SA 2.5 India — chosen over OSM's own
`admin_level=6` ward polygons because OSM's ward coverage there is partial and
ambiguous, while the DataMeet file is the canonical civic-data source).

The join can't be a naive `intersects` between segment geometries and ward
polygons: a segment that crosses a ward boundary would match *both* wards,
duplicating it. The fix is to join on a **representative point** of each
segment instead of the segment itself — `shapely`'s `representative_point()`
(not `centroid()`; a concave line's centroid can fall outside the geometry
entirely) guarantees a point inside the geometry, and a single point is in at
most one polygon *for a clean polygon layer*. In practice, "clean" needed a
caveat: two individually-`is_valid` BBMP wards (Hoysala Nagar / New
Bayappanahalli) turned out to overlap along their shared edge — a real
topology error in the source data, distinct from the ~6 wards that fail
`is_valid` outright. One segment's representative point landed in that overlap
sliver and matched both wards, breaking the "exactly one match" assumption the
whole join design depends on. The fix keeps the design's intent (deterministic,
exactly one ward per segment) by dropping duplicate matches and keeping the
first: `joined[~joined.index.duplicated(keep="first")]`.

Orphaned segments (representative point outside every ward — rare inside
Bengaluru) get empty `ward_id`/`ward_name` and `NaN` `ward_lgd_code` rather than
being dropped, and the coverage report surfaces the orphan rate so it's visible
rather than silently lossy.

## 6. Bugs caught during development, and what they revealed

Documenting these because each one taught something about the actual failure
mode — not "a bug happened," but *why* the obvious-looking code was wrong.

**The `osmid` MultiIndex issue.** OSMnx 2.x changed how it returns OSM
identifiers: features come back with a MultiIndex like `('element', 'id')`
instead of a top-level `osmid` column (older versions' convention). The
original loader assumed the column, so every `osmid` silently came back
`None` after writing to GeoJSON — not an exception, just quietly wrong data
that would only surface much later, whenever something tried to key off
`osmid`. `_extract_osm_ids` now tries three sources in order (top-level column,
MultiIndex level, single named index) before giving up. The lesson: a
third-party library's data-shape *convention* is not part of its stable API,
even when the convention feels obvious.

**The `.gitignore` bug.** `models/`, `artifacts/`, and `checkpoints/` were
written without a leading slash, intending to ignore top-level output
directories. Without the slash, git's pattern matches a directory with that
name *at any depth* — which silently excluded `src/paveiq/models/` (the real
source package, not a build artifact) from version control since the very
first scaffold commit. Nobody noticed because the files still existed on disk
locally; `git status` only complains about untracked files it can see, and
`.gitignore` makes git not see them. It surfaced only when `git ls-files
src/paveiq/models/` came back empty during an unrelated commit review. Fixed
by anchoring to the repo root (`/models/`), and — once a demo-data exception
was needed later — `/artifacts/*` + `!/artifacts/demo/` (a bare `/artifacts/`
would have excluded the directory itself, and git won't look inside an
excluded directory even to honor a negation pattern for a subdirectory of it).

**The fiona → pyogrio migration.** A Streamlit Cloud deploy failed because
`fiona` links against a system GDAL install that isn't available in that
environment; `pyogrio` ships GDAL as a prebuilt wheel and needs no system
dependency. The fix required **zero code changes** — nothing in the codebase
ever imported `fiona` directly or set `engine="fiona"`; every `geopandas` call
was already engine-agnostic, and geopandas has preferred `pyogrio` over `fiona`
by default since 0.14 when both are installed. The only change was
`requirements.txt`. The lesson: depending on a library's *default* engine
selection, rather than pinning one explicitly, is what made this a one-line
fix instead of a code migration.

**The `surface_quality` float-mapping bug.** Building a human-readable map
tooltip label (`{1.0: "Paved", 0.6: "Compacted", 0.2: "Unpaved"}`), a plain
`Series.map(dict)` against the raw `surface_quality` column silently returned
`NaN` for every row that wasn't exactly `1.0`. The column is stored as
`float32`; `float32(0.6) == 0.6` (the Python `float` literal) evaluates `True`
under numpy's elementwise comparison, but **dict lookups use hashing**, and
`hash(np.float32(0.6))` doesn't necessarily equal `hash(0.6)` even when `==`
succeeds — so the "obviously correct" `==`-equivalence check that would catch
most bugs like this didn't catch this one, because dict `.map()` never calls
`==` in the way that check exercised. It surfaced by comparing the mapped
output's `value_counts()` against the raw column's own `value_counts()` and
noticing `"Compacted"`/`"Unpaved"` had vanished — a check for the shape of
"nothing's obviously broken" wouldn't have caught it, only a check against the
*actual known distribution* did. Fixed by casting to `float64` before mapping.

**The what-if rank-sentence bug.** The what-if panel's "this change would move
ward X from rank N to rank M" sentence initially reported "would not change
rank" for a toggle that, by the app's own earlier-verified numbers, should have
moved a ward from the worst position to roughly the middle of the pack. The
cause: `apply_hypothetical` only mutates the *raw feature* columns (e.g.
`sidewalk_presence`); it deliberately doesn't recompute `score` itself (that's
the caller's job, via the scorer). The rank-comparison code merged this
modified-but-not-rescored subset back into a full copy of the table and
computed ranks from it — comparing against a `score` column that still held
the *pre-toggle* value. The bug produced no exception and a plausible-looking
sentence; it was only caught by checking the output against a number already
known from earlier manual verification, not by trusting that "it ran without
error" meant "it's correct." Fixed by recomputing `score` via `scorer.predict()`
on the modified subset before merging it into the table used for ranking.

## 7. Dashboard design choices

**Dark theme, electric-cyan accent.** Chosen for a "data/tech" read rather than
default Streamlit's light-blue look, but the palette wasn't eyeballed — every
color (`#0e1117` background, `#00d9ff` accent, `#0ca30c`/`#e66767` good/bad
status) was checked with a WCAG-contrast + OKLCH validator against the actual
rendered background, not a generic default surface. The leaderboard's score
gradient specifically blends matplotlib's `RdYlGn` 40% toward the background
color rather than using it raw: unblended `RdYlGn`'s yellow midpoint is light
enough that white text loses almost all contrast against it on a dark page —
the blend keeps the same red→yellow→green *direction* (visual consistency with
the map) while keeping text readable at every point on the gradient.

**Height = inverse score.** The 3D map extrudes each footpath into a ribbon
polygon (buffered from the line geometry — pydeck extrudes polygons, not lines)
whose height is `100 - score`, not `score` itself: worse footpaths spike up
taller. This is a deliberate inversion of the "taller is better" convention a
bar-chart reader might expect, chosen because it makes problems visually
self-flag without needing a legend to explain a positive-height-is-good
mapping — it directly serves the project's stated "worst-off segments surfaced
first" advocacy goal. (The ribbon geometry itself also went through a real
optimization pass after a production deploy hit a browser WebSocket
message-size limit sending the full per-segment payload for ~22k segments —
line simplification, mitre joins instead of round, coordinate-precision
rounding, and trimming GeoJSON properties to only what the tooltip needs
together cut that payload by 60% with no visible change in how the map looks.)

**`CARTO_DARK` basemap.** Chosen specifically because it's free (no Mapbox
account/token needed) and matches the project's `map_provider="carto"` setup
already in place for the light basemap — swapping to a dark tile set for a dark
dashboard theme was a one-parameter change (`pdk.map_styles.CARTO_DARK`)
precisely because the basemap provider was already CARTO, not Mapbox.

---

*This document reflects the state of the codebase as of commit `3dd972f`. See
the module docstrings in `src/paveiq/` for the most current, code-adjacent
version of this reasoning — this file is the narrative; the docstrings are the
source of truth if they ever diverge.*

*See the [README's Acknowledgements](../README.md#acknowledgements) for project authorship.*

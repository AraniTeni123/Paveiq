# PaveIQ

[![CI](https://github.com/AraniTeni123/Paveiq/actions/workflows/ci.yml/badge.svg)](https://github.com/AraniTeni123/Paveiq/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

## 🔗 Live Demo

[![Live Demo](https://img.shields.io/badge/Live%20Demo-paveiq--bengaluru.streamlit.app-00d9ff?style=for-the-badge&logo=streamlit&logoColor=white)](https://paveiq-bengaluru.streamlit.app)

**Live Demo → [paveiq-bengaluru.streamlit.app](https://paveiq-bengaluru.streamlit.app)**

**Predictive footpath health scoring for Bengaluru, built with geospatial data and machine learning.**

PaveIQ is a project for [Bengawalk](https://bengawalk.org), a Bengaluru-based urban walkability organization. The goal is to score and predict the condition of footpaths (sidewalks) across the city using open geospatial data, street imagery features, and ML — so that advocacy, repair prioritization, and citizen reporting can be data-driven.

For the *why* behind the design decisions below — OSM data-source choices, feature-engineering tradeoffs, the heuristic-vs-trained-model call, the ward-join methodology, real bugs caught along the way, and the dashboard's visual design — see **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

## Why this exists

Bengaluru's footpaths are uneven, broken, encroached, or simply missing. Manual surveys don't scale across thousands of kilometers of road network. PaveIQ builds a **footpath health score** from publicly available geospatial signals (OpenStreetMap, satellite/raster features, BBMP / ward boundaries, citizen reports) so that:

1. **Worst-off segments are surfaced first** — the model flags where footpath quality is most likely to be poor.
2. **Surveys can be targeted** — limited field-verification budget is spent where it matters.
3. **Trends are trackable** — re-scoring on new data shows whether interventions are working.

## Project layout

```
paveiq/
├── src/paveiq/         # Main pipeline package
│   ├── data_ingestion/ # Pull and normalize raw geospatial data
│   ├── features/       # Feature engineering over rasters, vectors, network
│   ├── models/         # Scorer training, evaluation, persistence
│   ├── scoring/        # Apply the scorer to new areas, produce maps
│   └── dashboard/      # Streamlit app: 3D map, ward leaderboard, what-if panel
├── data/
│   ├── raw/            # Original downloaded data (gitignored)
│   ├── processed/      # Pipeline output (gitignored)
│   └── demo/           # Bundled fixed snapshot for the deployed dashboard (tracked — see Demo dataset)
├── artifacts/          # Serialized scorer artifacts (gitignored)
│   └── demo/           # Bundled scorer artifact for the deployed dashboard (tracked — see Demo dataset)
├── notebooks/          # Exploration, ad-hoc analysis
├── tests/              # Unit and integration tests
├── requirements.txt
└── README.md
```

## Pipeline overview

The pipeline runs in stages, each reading the previous stage's output and writing to `data/processed/`.

1. **Data ingestion** — fetch OpenStreetMap footpath geometries and BBMP ward boundaries. Normalize to a common CRS and schema.
2. **Feature engineering** — for each footpath segment, compute features: length, width (where mappable), highway-likelihood, surface quality, sidewalk presence.
3. **Ward join** — spatial-join each segment to its BBMP ward via a representative-point join.
4. **Scoring model** — no labeled citizen-report/field-verification data exists yet, so scoring currently runs on a transparent heuristic (a weighted combination of the engineered features, see `models/heuristic.py`) rather than a trained regression/classifier. It's built behind the same `predict(df) -> score` interface a trained model would expose, so swapping one in once labels exist requires no changes to the scoring stage or the dashboard.
5. **Scoring** — apply the scorer to the full city network; output a scored Parquet.
6. **Dashboard** — a Streamlit app (3D pydeck map, ward leaderboard, what-if simulator) reads the scored Parquet directly.

## Status

The ingestion → features → ward-join → scoring pipeline runs end-to-end, and a Streamlit dashboard sits on top of it. No labeled footpath-condition data (citizen reports / field verification) exists yet, so scoring uses a transparent heuristic rather than a trained model — see `models/heuristic.py` for the interface a future trained model would drop into.

## Setup

### 1. Clone and enter the project

```bash
git clone <repo-url> paveiq
cd paveiq
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Some geospatial libraries (notably `rasterio`) link against GDAL, though both `rasterio` and `pyogrio` (geopandas' I/O engine) ship GDAL as a prebuilt wheel, so a plain `pip install` works without a system GDAL install — this is also why the project uses `pyogrio` rather than `fiona`, which needs one. On macOS, if you still hit install errors, the conda-forge builds are a fallback:

```bash
conda install -c conda-forge geopandas rasterio pyogrio shapely pyproj
pip install -r requirements.txt   # for the ML and utility packages
```

### 4. Verify

```bash
python -c "import geopandas, shapely, rasterio, sklearn; print('OK')"
pytest
```

## Running the pipeline

Once the stages are implemented:

```bash
# Stage 1: fetch OSM footpath data (network access required).
# OSMnx is an optional dep; install it first:  pip install -e ".[osm]"
python -m paveiq.data_ingestion.osm_loader --place "Bengaluru, India"
# ...or a small bbox for a fast first run:
python -m paveiq.data_ingestion.osm_loader --bbox 12.93 77.55 12.99 77.65

# Stage 2: per-segment features.
python -m paveiq.features.build_features

# Stage 3: spatial-join to BBMP wards (downloads the
# DataMeet 2022 ward GeoJSON on first run, then caches it).
python -m paveiq.data_ingestion.ward_boundaries

# Stage 4: build the scorer artifact (heuristic for now — see Status above).
python -m paveiq.models.train

# Stage 5: apply the scorer to the full network.
python -m paveiq.scoring.score_city

# Stage 6: explore the results in the dashboard.
# streamlit/pydeck are an optional dep: pip install -e ".[dashboard]"
streamlit run src/paveiq/dashboard/app.py
```

`data/raw/` gets the OSM GeoJSON and the cached BBMP wards; `data/processed/` gets the
per-segment feature Parquet, the `*_with_wards.parquet`, and the `*_scored.parquet`;
`artifacts/` gets the scorer artifact.

## Demo dataset

`data/raw/`, `data/processed/`, and `artifacts/` are gitignored — they're pipeline working
directories, regenerated by running the stages above. That means a fresh deploy (e.g.
Streamlit Cloud) has nothing in `data/processed/` or `artifacts/` for the dashboard to read.

`data/demo/` and `artifacts/demo/` are deliberate exceptions: `data/demo/` holds one committed
file, `*_features_with_wards_scored.parquet` — the final scored output of a real run of the
full pipeline through stage 5, for the Koramangala bbox (`12.93 77.55 12.99 77.65`);
`artifacts/demo/` holds the matching `heuristic_scorer_v1.json` scorer used to produce it. The
dashboard's data loader (`dashboard/data.py`) prefers the real `data/processed/*_scored.parquet`
and `artifacts/*_scorer*.json` and only falls back to the `demo/` copies when either isn't
found, so local dev running the full pipeline is unaffected — the deployed app (data and
what-if panel both) just has something to show.

**This is a fixed snapshot of one neighborhood, not live-updating data.** It does not reflect
the current state of Bengaluru's footpaths, and does not update as the pipeline or scoring
model changes — regenerate it by re-running stages 2-5 above and copying the new
`*_scored.parquet` into `data/demo/` if it needs to change.

## Data sources (planned)

- **OpenStreetMap** — footpath geometries, surface tags, width, accessibility tags.
- **BBMP ward boundaries** — administrative overlay. Sourced from
  [DataMeet / Municipal_Spatial_Data](https://github.com/datameet/Municipal_Spatial_Data/tree/master/Bangalore)
  (CC-BY-SA 2.5 India), 2022 delimitation (243 wards). Cached
  locally at `data/raw/bbmp_wards_2022.geojson` on first run.
- **Sentinel-2 / Landsat rasters** — vegetation, built-up fraction.
- **Bengawalk citizen reports** — ground-truth footpath condition labels.

## Contributing

Issues and pull requests welcome. For larger changes, open an issue first to discuss scope.

## License

TBD — to be decided with Bengawalk.

## Acknowledgements

The project concept, research direction, and all design decisions and parameters are by Arani Mahapatra. Implementation was assisted by Claude Code and MiniMax (via Ollama) as AI coding assistants.

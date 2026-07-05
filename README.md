# PaveIQ

**Predictive footpath health scoring for Bengaluru, built with geospatial data and machine learning.**

PaveIQ is a project for [Bengawalk](https://bengawalk.org), a Bengaluru-based urban walkability organization. The goal is to score and predict the condition of footpaths (sidewalks) across the city using open geospatial data, street imagery features, and ML — so that advocacy, repair prioritization, and citizen reporting can be data-driven.

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
│   ├── models/         # Training, evaluation, persistence
│   └── scoring/        # Apply model to new areas, produce maps
├── data/
│   ├── raw/            # Original downloaded data (gitignored)
│   └── processed/      # Pipeline output (gitignored)
├── notebooks/          # Exploration, ad-hoc analysis
├── tests/              # Unit and integration tests
├── requirements.txt
└── README.md
```

## Pipeline overview

The pipeline has four stages, run in order. Each stage reads from the previous stage's output and writes to `data/processed/`.

1. **Data ingestion** — fetch OpenStreetMap footpath geometries, ward/BBMP boundaries, raster layers (e.g., NDVI, impervious-surface), and any available citizen reports. Normalize to a common CRS and schema.
2. **Feature engineering** — for each footpath segment, compute features: length, width (where mappable), nearby amenity density, road class, vegetation proximity, surface type, etc.
3. **Modeling** — train a regression or classifier on labeled segments (citizen reports + field verification) to predict a 0–100 health score.
4. **Scoring** — apply the trained model to the full city network; output a scored GeoPackage / GeoJSON for use in QGIS, kepler.gl, or a simple web map.

## Status

Early stage. The directory structure and package skeleton are in place; data sources, schemas, and the first baseline model are next.

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

Some geospatial libraries (notably `rasterio` and `fiona`) link against GDAL. On macOS, if you hit install errors, the easiest path is to use the conda-forge builds:

```bash
conda install -c conda-forge geopandas rasterio fiona shapely pyproj
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

# Stages 2-4 (not yet implemented):
python -m paveiq.features.build_features
python -m paveiq.models.train
python -m paveiq.scoring.score_city
```

`data/raw/` gets the OSM GeoJSON; the rest of the pipeline writes to `data/processed/`.

## Data sources (planned)

- **OpenStreetMap** — footpath geometries, surface tags, width, accessibility tags.
- **BBMP ward boundaries** — administrative overlay.
- **Sentinel-2 / Landsat rasters** — vegetation, built-up fraction.
- **Bengawalk citizen reports** — ground-truth footpath condition labels.

## Contributing

Issues and pull requests welcome. For larger changes, open an issue first to discuss scope.

## License

TBD — to be decided with Bengawalk.

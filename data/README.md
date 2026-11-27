# Data

Raw datasets are **not distributed** in this repository (size and licensing).
Place the required inputs in this directory (or edit the path at the top of each script in `../src/`).

## Required inputs

### Predictand — Korea-mean series (`year` + `korea_nday`, identical structure)

| file | description | source |
|---|---|---|
| `ERA5_DJF_EC_index_1979-2024.nc` | DJF extreme-cold-day count, Korea cos-weighted mean | ERA5 |
| `ERA5_DJF_EW_index_1979-2024.nc` | DJF extreme-warm-day count, Korea cos-weighted mean | ERA5 |

### ERA5 monthly fields — selected predictors

| field | role |
|---|---|
| **DJF T2M** | EC α, EW α (L0, concurrent) |
| **DJF T850** | EW γ (L0, concurrent) |
| **NOV T850** | EW β (L1, observed lead) |
| **SEP precipitation** | EC β (L3, observed lead) |
| **OCT precipitation** | EC γ (L2, observed lead) |

Source: ERA5 — Copernicus Climate Data Store (CDS).

### C3S multi-model-ensemble forecast — concurrent (L0) predictors

For the L0 predictors **DJF T2M** and **DJF T850**, two forms are required (the probabilistic/tercile forecast needs the ensemble):

| file | description |
|---|---|
| `c3s_mme_L0_djf.csv` | **MME (model-mean)** L0 predictor series — deterministic / hybrid forecast |
| `c3s_ensemble_L0_djf.csv` | **full per-member ensemble** L0 predictor values — tercile / probabilistic forecast |

Source: C3S seasonal multi-model forecasts — Copernicus CDS.

### Produced by the code (not uploaded)

`predictor_accepted_era5.npz` — from `src/01_make_predictor_index.py`.

> Note: the shared pool-builder `01_make_predictor_index.py` screens a broader set of ERA5 monthly fields and also references the summer `ERA5_JA_CDHD_index_1979-2024.nc`; the five fields above are what the **final EC/EW models** use.

## Data sources

- **ERA5 reanalysis** — Copernicus Climate Data Store: https://cds.climate.copernicus.eu
- **C3S seasonal forecasts (multi-model)** — Copernicus CDS.

Download / preprocessing scripts are project-internal; please contact the authors for the exact intermediate files.

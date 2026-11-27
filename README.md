# SJU-Hybrid · Winter Extreme (Nonlinear / CatBoost)

Machine-learning (gradient-boosted decision trees, **CatBoost**) seasonal prediction of **Korean winter extreme-cold days (EC) and extreme-warm days (EW)** over the December–January–February (DJF) season.

This repository contains the **nonlinear (CatBoost) counterpart** of the statistical–dynamical model developed in the Korea Meteorological Administration (KMA) R&D project *“AI-based research on climate-extreme variability and prediction”* (Sejong University component). It uses the **same physically-screened predictor pool** as the linear model but replaces multiple linear regression with a **gradient-boosted tree estimator**, producing **tercile (below / near / above-normal) probabilistic forecasts**.

## Overview

- **Predictands:** DJF Korean extreme-cold-day count (**EC**) and extreme-warm-day count (**EW**), from an ERA5-based percentile index.
- **Predictors:** identical physically-motivated, skill-screened teleconnection pool as the linear model (see the sibling `…-linear` repository).
- **Estimator:** **CatBoost** (symmetric gradient-boosted trees). CatBoost is chosen for small-sample robustness (strong regularization via oblivious trees + ordered boosting) over XGBoost/LightGBM.
- **Output:** per-member hybrid forecast (observed lead predictors + C3S-MME L0), aggregated to tercile-category probabilities.

## Method (pipeline)

| step | script | description |
|---|---|---|
| 1 | `src/00_corr_leads.py` | Detrended correlation of the EC/EW index against global climate fields at leads 0–3 months (teleconnection diagnosis). |
| 2 | `src/01_make_predictor_index.py` | Predictor-pool construction: significance ∩ dynamical-skill mask → 8-connected clustering (≥100 cells) → area-mean index → collinearity screening + SH exclusion. *(Shared with the linear model.)* |
| 3 | `src/02_forecast_catboost.py` | Leave-one-out CatBoost regression on the pooled predictors; per-ensemble-member prediction (observed leads + C3S-MME L0), categorized into observed terciles → below/near/above-normal probabilities. |

## Selected predictors

- **EC (extreme-cold days):** DJF 2-m temperature over Siberia (L0) · September precipitation over the North Pacific (L3) · October precipitation over western Russia (L2)
- **EW (extreme-warm days):** DJF 2-m temperature over mid-latitude Eurasia (L0) · November 850-hPa temperature over the tropical western Pacific (L1) · DJF 850-hPa temperature over the Bering/Alaska region (L0)

## Repository structure

```
.
├── src/
│   ├── 00_corr_leads.py           # teleconnection correlation maps (predictand vs fields, leads 0–3)
│   ├── 01_make_predictor_index.py # predictor-pool construction (shared with the linear model)
│   └── 02_forecast_catboost.py    # CatBoost tercile probabilistic forecast (EC_day | EW_day)
├── data/                          # input data (not included — see data/README.md)
├── requirements.txt
└── README.md
```

## Data

Raw datasets are **not included** (size / licensing). Place inputs under `./data/` (or set the path at the top of each script). Required inputs include the ERA5-based EC/EW index, ERA5 monthly fields, the screened predictor pool (`predictor_accepted_era5.npz`), and the **per-member C3S forecast predictor table** for the ensemble forecast. See `data/README.md`.

## Usage

```bash
pip install -r requirements.txt
# place inputs under ./data  (see data/README.md)
python src/00_corr_leads.py
python src/01_make_predictor_index.py
python src/02_forecast_catboost.py EC_day   # EC_day | EW_day
```

## Authors

Jee-Hoon Jeong (정지훈)¹, Gyu-Ri Lee (이규리)¹
¹ Sejong University, Republic of Korea

## Funding

This research was supported by the **Korea Meteorological Administration Research and Development Program** under Grant **RS-2025-02307979**.

## Citation

If you use this code, please cite the associated project report / manuscript (Jeong, Lee et al., 2026) and acknowledge the funding grant above.

## Legal Notice

This repository contains original research code developed for academic and scientific purposes.

Any use, reproduction, modification, distribution, or derivative work based on this code is strictly prohibited without prior written permission from the authors.

If this code is used in any form of academic, research, or commercial work, proper citation and explicit acknowledgment of all original authors and affiliated institutions is mandatory.

Unauthorized use or redistribution without permission may result in legal consequences.

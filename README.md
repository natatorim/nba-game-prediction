# NBA Game Outcome Prediction
**COMP375: Machine Learning — Final Project**  
**Natalia Nguyen**

## Overview
Machine learning pipeline to predict NBA game outcomes using 10 years of game data (9,528 observations, 2015–2025). Two predictive tasks compared across 4 models, with a focus on proper time-series evaluation methodology.

## Predictive Questions
1. **Classification:** Can we predict whether the home team will win? (Binary: HOME_WIN)
2. **Regression:** Can we predict the points differential? (Continuous: ACTUAL_PTS_DIFF)

## Key Findings
- Linear models outperformed Random Forest on both tasks despite being simpler — the feature set lacked enough signal for ensemble complexity to be useful, and both Random Forest models showed persistent overfitting
- Models trained on COVID bubble seasons (2020–2021) failed completely on 2022 regular season games (Ridge R² collapsed from ~0.09 to 0.019) — the model had encoded home court advantage and travel assumptions that broke down during the pandemic
- Random splitting inflated MAE by over 1 point compared to proper time-series splitting, demonstrating why evaluation methodology matters as much as model choice
- A sliding window evaluation (train 2 years, test 1 year) revealed that classification performance is stable across all eras (AUC 0.61–0.68) while regression is sensitive to playstyle shifts and abnormal seasons

## Models

| Task | Baseline | Complex |
|---|---|---|
| Classification | Logistic Regression | Random Forest Classifier |
| Regression | Ridge Regression | Random Forest Regressor |

## Results

| Model | Metric | Train | Val | Test |
|---|---|---|---|---|
| Logistic Regression | AUC | 0.655 | 0.649 | 0.660 |
| Random Forest Classifier | AUC | 0.657 | 0.633 | 0.654 |
| Ridge Regression | R² | 0.094 | 0.094 | 0.107 |
| Ridge Regression | MAE | — | — | 11.97 pts |
| Random Forest Regressor | R² | 0.094 | 0.072 | 0.078 |
| Random Forest Regressor | MAE | — | — | 12.15 pts |

## Features
All features are pre-game rolling averages — no data leakage:

| Feature | Description |
|---|---|
| PTS_DIFF | Rolling points differential (home − away) |
| REB_DIFF | Rolling rebound differential |
| AST_DIFF | Rolling assist differential |
| TOV_DIFF | Rolling turnover differential |
| EFG_DIFF | Rolling effective field goal % differential |
| TRAVEL_DIST | Away team travel distance to game (km) |

Three rolling window sizes evaluated: Last-5, Last-20, Last-50 games.  
**Last-20 selected** as primary window — best balance of recency and stability.

## Data Split
- **70% training** — chronological, older games
- **20% validation** — used for hyperparameter tuning only
- **10% test** — touched once at the very end

Random shuffling was explicitly avoided — future games must never appear in training data.

## Hyperparameters

| Model | Parameters |
|---|---|
| Logistic Regression | C=1.0, max_iter=100, solver=lbfgs, penalty=l2 |
| Random Forest Classifier | n_estimators=100, max_depth=3, min_samples_leaf=10 |
| Ridge Regression | alpha=1.0 |
| Random Forest Regressor | n_estimators=100, max_depth=3, min_samples_leaf=10 |

Random Forest parameters were tuned on the validation set to reduce overfitting.  
All models use StandardScaler fitted on training data only.

## Reproducing the Data
Data collected from the NBA Stats API using the `nba_api` Python library.

Run `ok.ipynb` three times, changing the rolling window parameter to 5, 20, and 50.  
Save outputs as:
- `window=5` → `nba_long_dataset.csv`
- `window=20` → `nba_long_dataset2.csv`
- `window=50` → `nba_long_dataset3.csv`

Then run `finalcode.ipynb` for the full modeling pipeline.

## Files
- `finalcode.ipynb` — complete modeling pipeline (data loading, splitting, training, evaluation, figures)
- `ok.ipynb` — data collection from NBA API
- `NBA_Final_Report_Natalia_Nguyen.pdf` — full project report

## Dependencies
```bash
pip install nba_api pandas scikit-learn matplotlib numpy
```

## Tools
Python, scikit-learn, pandas, NumPy, matplotlib, nba_api

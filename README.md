# Heart Disease Risk Analytics

AI-powered predictive analytics system for healthcare risk analysis. Trains,
evaluates, and serves a machine learning model that estimates a patient's
risk of heart disease from clinical attributes, with model outputs and KPIs
exported for a Power BI dashboard.

**Stack:** Python, scikit-learn, Flask, Power BI

## What this project does

- Feature engineering over 13 clinical patient attributes (age, chest pain
  type, cholesterol, ECG results, etc.), including age-risk-band bucketing,
  missing-value imputation, and encoding/scaling.
- Model selection and cross-validation: a plain Logistic Regression baseline
  vs. a StratifiedKFold-cross-validated, GridSearchCV-tuned Random Forest.
- A Flask REST API (`/predict`, `/health`, `/model-info`) that serves live
  risk scores from the trained pipeline.
- Power BI-ready exports (KPI workbook + patient-level predictions table)
  for a non-technical reviewer dashboard.

## Architecture

```
                     ┌─────────────────────────┐
                     │   UCI Heart Disease      │
                     │   dataset (ucimlrepo)    │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  src/data.py             │
                     │  fetch + cache + binarize│
                     │  target                  │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  src/features.py         │
                     │  age bands, impute,      │
                     │  encode, scale           │
                     │  (sklearn ColumnTransf.) │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  src/train.py            │
                     │  baseline LogReg  vs.     │
                     │  tuned RandomForest       │
                     │  (StratifiedKFold + Grid- │
                     │   SearchCV)               │
                     └──────┬─────────────┬──────┘
                            │             │
                            ▼             ▼
             ┌───────────────────┐  ┌─────────────────────────┐
             │ models/            │  │ exports/                 │
             │ risk_model.pkl     │  │ model_kpis.xlsx           │
             │ model_metadata.json│  │ predictions.csv           │
             └─────────┬──────────┘  └────────────┬─────────────┘
                       │                           │
                       ▼                           ▼
             ┌───────────────────┐      ┌──────────────────────┐
             │ app.py (Flask API)│      │ Power BI dashboard     │
             │ /predict           │      │ (see DASHBOARD.md)     │
             │ /health            │      │ KPI cards, risk donut, │
             │ /model-info        │      │ feature importance,   │
             │                    │      │ patient drill-down     │
             └───────────────────┘      └──────────────────────┘
```

## Dataset

**UCI Heart Disease dataset (Cleveland subset)** — 303 patient records, 13
clinical attributes, binary target (heart disease presence). Fetched
programmatically via [`ucimlrepo`](https://pypi.org/project/ucimlrepo/),
no login or API key required, and cached locally at
`data/heart_disease.csv` after the first run.

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart
> Disease [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C52P4X

The 13 raw features: `age`, `sex`, `cp` (chest pain type), `trestbps`
(resting blood pressure), `chol` (cholesterol), `fbs` (fasting blood
sugar), `restecg` (resting ECG), `thalach` (max heart rate achieved),
`exang` (exercise-induced angina), `oldpeak` (ST depression), `slope`,
`ca` (number of major vessels), `thal` (thalassemia). The original 5-class
severity target (`num`, 0-4) is collapsed into a binary indicator
(0 = absent, 1 = present), the standard framing for this dataset.

## Setup

```bash
python3.10 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

```bash
python -m src.train
```

This fetches (or loads the cached) dataset, engineers features, trains the
baseline and improved models, evaluates both on the same held-out test set,
saves the winning pipeline to `models/risk_model.pkl`, and writes the
Power BI exports (`exports/model_kpis.xlsx`, `exports/predictions.csv`).

## Run the API

```bash
python app.py
```

### `GET /health`

```json
{"status": "ok"}
```

### `GET /model-info`

Returns model type, tuned hyperparameters, the raw feature list, and the
training metrics (baseline vs. improved).

### `POST /predict`

Request body — all 13 raw attributes (`ca`/`thal` may be omitted or `null`,
since the pipeline imputes them the same way it does at training time):

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0.0, "thal": 6.0
  }'
```

```json
{
  "risk_score": 0.3046,
  "risk_category": "Low",
  "prediction": 0,
  "prediction_label": "heart disease absent"
}
```

## Model performance (actual measured results)

Evaluated on the same held-out 20% test split (61 patients) for both
models, so the comparison is apples-to-apples.

| Metric    | Baseline (Logistic Regression) | Improved (Random Forest, tuned) |
|-----------|:-------------------------------:|:---------------------------------:|
| Accuracy  | 86.89%                          | 88.52%                            |
| Precision | 81.25%                          | 83.87%                            |
| Recall    | 92.86%                          | 92.86%                            |
| F1        | 0.8667                          | 0.8814                            |
| ROC-AUC   | 0.9578                          | 0.9589                            |

**Measured accuracy lift over baseline: +1.89%** (86.89% → 88.52%), from a
Random Forest tuned via `GridSearchCV` over `n_estimators`, `max_depth`,
`min_samples_split`, and `min_samples_leaf` with `StratifiedKFold(5)`
cross-validation, scored on accuracy. Best CV accuracy: see
`models/model_metadata.json` for the full `best_params`.

**Honesty note on resume framing:** the original scope for this project
targeted "~20% higher accuracy than baseline." The actual measured lift on
this dataset is a more modest +1.89%. A 10-fold cross-validation sweep
across the full 303-row dataset backs this up — Logistic Regression
(84.46% mean accuracy) is a genuinely strong, close-to-linear baseline for
this classic, well-studied dataset, and Random Forest (82.82%) and
Gradient Boosting (79.49%) don't reliably beat it at this sample size. This
is a known property of the Cleveland Heart Disease dataset, not a bug in
the pipeline. **A resume bullet describing this project should report the
actual measured lift (or describe the improvement qualitatively — better
precision, comparable recall, higher ROC-AUC) rather than a fixed 20%
target.**

**Top features by importance** (from the tuned Random Forest):
`thal` (fixed defect), `ca` (number of major vessels), `thal` (reversible
defect), `cp` (asymptomatic chest pain), `thalach` (max heart rate). Full
ranking in `exports/model_kpis.xlsx` → *Feature Importances*.

## Project structure

```
healthcare-risk-analytics/
├── app.py                     # Flask REST API
├── requirements.txt
├── DASHBOARD.md                # Power BI build spec
├── LICENSE
├── data/
│   └── heart_disease.csv       # cached dataset (fetched via ucimlrepo)
├── notebooks/
│   └── 01_eda.ipynb            # exploratory data analysis
├── src/
│   ├── data.py                 # fetch/cache/binarize
│   ├── features.py             # feature engineering pipeline
│   └── train.py                # baseline + improved model, exports
├── models/
│   ├── risk_model.pkl           # persisted sklearn Pipeline (preprocess+model)
│   └── model_metadata.json
├── exports/
│   ├── model_kpis.xlsx          # Power BI-ready KPI workbook
│   └── predictions.csv          # patient-level predictions
└── tests/
    ├── test_features.py
    └── test_app.py
```

## Tests

```bash
pytest tests/ -v
```

Covers the feature-engineering pipeline (age banding, missing-value
imputation for `ca`/`thal`, output shape) and the Flask endpoints (health,
model info, valid/invalid predictions).

## Power BI dashboard

Power BI Desktop is a separate Windows application and isn't something a
script can generate — see [`DASHBOARD.md`](DASHBOARD.md) for the exact
build spec (visuals, field mappings, ~15-20 min build time).

**Screenshots:** *(add exported dashboard screenshots here after building
it in Power BI Desktop — e.g. `docs/dashboard-overview.png`,
`docs/dashboard-patient-drilldown.png`)*

## Repository integrity

> Do not fabricate git commit dates or otherwise alter version-control
> metadata to make this repository appear older than it is. Commit
> normally, at the time the work is actually done, with accurate
> timestamps.

## License

MIT — see [LICENSE](LICENSE).

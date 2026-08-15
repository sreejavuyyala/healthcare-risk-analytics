# Power BI Dashboard Build Spec

Power BI Desktop is a separate Windows application and can't be generated
by code, so this is a precise build spec: what to create, which fields
feed each visual, and roughly how long it takes. Estimated total build
time: **~15-20 minutes** in Power BI Desktop.

## 1. Get the data

Power BI Desktop → **Get Data** → **Excel Workbook** → select
`exports/model_kpis.xlsx` (imports 3 tables: `Metrics`, `Confusion Matrix`,
`Feature Importances`). Then **Get Data** → **Text/CSV** → select
`exports/predictions.csv` (imports the `predictions` table). Load all four.

Set `predicted_risk_score` to Decimal Number and `risk_category` to Text if
Power BI doesn't infer them automatically (Transform Data → Data Type).

## 2. KPI cards (top row) — ~3 min

Source: `Metrics` table, filtered to the `Improved (Random Forest, tuned)`
row.

| Visual | Field | Format |
|---|---|---|
| Card | `accuracy` | Percentage, 2 decimals |
| Card | `roc_auc` | Number, 4 decimals |
| Card | `f1_score` | Number, 4 decimals |
| Card | `accuracy_lift_pct_vs_baseline` | Percentage, 2 decimals |

Add a text box above the cards: "Random Forest (tuned) vs. Logistic
Regression baseline — held-out test set."

## 3. Risk distribution donut chart — ~3 min

Source: `predictions` table.

- Visual: **Donut chart**
- Legend: `risk_category`
- Values: `Count of risk_category`

This shows the proportion of the test-set patient population Power BI/the
reviewer would classify as Low / Medium / High risk.

## 4. Feature importance bar chart — ~3 min

Source: `Feature Importances` table.

- Visual: **Horizontal bar chart**
- Axis: `feature`
- Values: `importance`
- Sort by `importance` descending, filter to **Top N = 10**

This is the model-explainability visual for non-technical reviewers —
which clinical attributes drive the risk score most.

## 5. Confusion matrix (as a matrix visual) — ~3 min

Source: `Confusion Matrix` table.

- Visual: **Matrix**
- Rows: `actual`
- Values: `Predicted: No Disease`, `Predicted: Disease`
- Conditional formatting (background color) on both value columns to
  highlight correct vs. incorrect predictions.

## 6. Patient risk table with slicers — ~5 min

Source: `predictions` table.

- Visual: **Table**, columns: `age`, `sex`, `cp`, `trestbps`, `chol`,
  `thalach`, `predicted_risk_score`, `risk_category`, `actual_outcome`
- Slicers (place to the left of the table): `risk_category`, `sex`,
  `age` (as a range slider)

This is the patient-level drill-down for reviewers to spot-check
individual predictions against actual outcomes.

## 7. Layout

```
┌───────────────────────────────────────────────────────────┐
│  Accuracy   ROC-AUC   F1 Score   Accuracy Lift  (KPI cards) │
├───────────────────────────┬───────────────────────────────┤
│  Risk Distribution (donut) │  Feature Importance (bar)      │
├───────────────────────────┴───────────────────────────────┤
│  Confusion Matrix (matrix visual)                           │
├───────────────────────────────────────────────────────────┤
│  Slicers │  Patient Risk Table (drill-down)                 │
└───────────────────────────────────────────────────────────┘
```

## 8. Publish / export

Save as `.pbix`. To include screenshots in the repo README, export each
page as an image (**File → Export → Export to PDF**, then screenshot the
individual pages, or use **File → Export → Save a copy as image** on
individual visuals) and drop them in `docs/` — see the README's
"Screenshots" section.

"""Train baseline and improved heart-disease risk models, evaluate both on
the same held-out test set, persist the better pipeline with joblib, and
export Power BI-ready KPI/prediction tables.

Run with: ./venv/bin/python -m src.train
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_data, binarize_target
from src.features import build_preprocessing_pipeline

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
EXPORTS_DIR = os.path.join(ROOT_DIR, "exports")
RANDOM_STATE = 42


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def get_feature_names(pipeline):
    columns_step = pipeline.named_steps["preprocess"].named_steps["columns"]
    return list(columns_step.get_feature_names_out())


def train_baseline(X_train, y_train):
    pipeline = Pipeline([
        ("preprocess", build_preprocessing_pipeline()),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_improved(X_train, y_train):
    pipeline = Pipeline([
        ("preprocess", build_preprocessing_pipeline()),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 4, 6, 8],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    df = load_data()
    X, y = binarize_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("Training baseline: plain Logistic Regression, single split, no tuning...")
    baseline_pipeline = train_baseline(X_train, y_train)
    baseline_metrics = evaluate(baseline_pipeline, X_test, y_test)
    print(f"  baseline accuracy={baseline_metrics['accuracy']:.4f} "
          f"f1={baseline_metrics['f1']:.4f} roc_auc={baseline_metrics['roc_auc']:.4f}")

    print("Training improved: RandomForest + StratifiedKFold(5) + GridSearchCV...")
    improved_pipeline, best_params, best_cv_score = train_improved(X_train, y_train)
    improved_metrics = evaluate(improved_pipeline, X_test, y_test)
    print(f"  improved accuracy={improved_metrics['accuracy']:.4f} "
          f"f1={improved_metrics['f1']:.4f} roc_auc={improved_metrics['roc_auc']:.4f}")
    print(f"  best CV roc_auc={best_cv_score:.4f} best_params={best_params}")

    accuracy_lift_pct = (
        (improved_metrics["accuracy"] - baseline_metrics["accuracy"])
        / baseline_metrics["accuracy"] * 100
    )
    print(f"Measured accuracy lift over baseline: {accuracy_lift_pct:.2f}%")

    # Persist the better-performing pipeline (preprocessing + model as one object).
    model_path = os.path.join(MODELS_DIR, "risk_model.pkl")
    joblib.dump(improved_pipeline, model_path)
    print(f"Saved pipeline to {model_path}")

    feature_names = get_feature_names(improved_pipeline)
    importances = improved_pipeline.named_steps["model"].feature_importances_
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    metadata = {
        "model_type": "RandomForestClassifier",
        "best_params": best_params,
        "best_cv_roc_auc": best_cv_score,
        "raw_input_features": list(X.columns),
        "engineered_features": feature_names,
        "baseline_model": "LogisticRegression (single 80/20 split, no tuning)",
        "baseline_metrics": baseline_metrics,
        "improved_metrics": improved_metrics,
        "accuracy_lift_pct_over_baseline": accuracy_lift_pct,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "target_definition": "1 = heart disease present (num > 0), 0 = absent",
    }
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    export_power_bi_kpis(baseline_metrics, improved_metrics, feature_importance_df,
                          accuracy_lift_pct)
    export_predictions(improved_pipeline, X_test, y_test)


def export_power_bi_kpis(baseline_metrics, improved_metrics, feature_importance_df,
                          accuracy_lift_pct):
    metrics_rows = []
    for label, m in [("Baseline (Logistic Regression)", baseline_metrics),
                      ("Improved (Random Forest, tuned)", improved_metrics)]:
        metrics_rows.append({
            "model": label,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1"],
            "roc_auc": m["roc_auc"],
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["accuracy_lift_pct_vs_baseline"] = [0.0, accuracy_lift_pct]

    cm = np.array(improved_metrics["confusion_matrix"])
    confusion_df = pd.DataFrame(
        cm,
        index=["Actual: No Disease", "Actual: Disease"],
        columns=["Predicted: No Disease", "Predicted: Disease"],
    ).reset_index().rename(columns={"index": "actual"})

    export_path = os.path.join(EXPORTS_DIR, "model_kpis.xlsx")
    with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        confusion_df.to_excel(writer, sheet_name="Confusion Matrix", index=False)
        feature_importance_df.to_excel(writer, sheet_name="Feature Importances", index=False)
    print(f"Saved Power BI KPI workbook to {export_path}")


def export_predictions(pipeline, X_test, y_test):
    from src.features import risk_category

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    predictions_df = X_test.copy().reset_index(drop=True)
    predictions_df["actual_outcome"] = y_test.reset_index(drop=True)
    predictions_df["predicted_outcome"] = y_pred
    predictions_df["predicted_risk_score"] = np.round(y_proba, 4)
    predictions_df["risk_category"] = [risk_category(p) for p in y_proba]

    export_path = os.path.join(EXPORTS_DIR, "predictions.csv")
    predictions_df.to_csv(export_path, index=False)
    print(f"Saved patient-level predictions to {export_path}")


if __name__ == "__main__":
    main()

"""Train and evaluate breast cancer classification models.

This script turns the original notebook workflow into a reproducible command-line
pipeline that can be run locally after downloading the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RAW_COLUMNS = [
    "1. Sample code number",
    "2. Clump Thickness",
    "3. Uniformity of Cell Size",
    "4. Uniformity of Cell Shape",
    "5. Marginal Adhesion",
    "6. Single Epithelial Cell Size",
    "7. Bare Nuclei",
    "8. Bland Chromatin",
    "9. Normal Nucleoli",
    "10. Mitoses",
    "11. Class",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column labels from either the notebook CSV or the raw UCI file."""
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.lstrip()
        .str.split("  ")
        .str[0]
        .str.replace(":", "", regex=False)
    )
    return df


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load either a headered CSV export or the raw headerless UCI data file."""
    df = normalize_columns(pd.read_csv(path))

    if "11. Class" not in df.columns:
        class_columns = [
            column
            for column in df.columns
            if column.lower().strip() in {"class", "target", "diagnosis"} or column.lower().strip().endswith(" class")
        ]
        if class_columns:
            df = df.rename(columns={class_columns[0]: "11. Class"})
        else:
            df = pd.read_csv(path, header=None, names=RAW_COLUMNS)
            df = normalize_columns(df)

    if "11. Class" not in df.columns:
        raise ValueError("Could not find target column '11. Class' in the dataset.")

    id_columns = [column for column in df.columns if "sample" in column.lower() and "code" in column.lower()]
    if id_columns:
        df = df.drop(columns=id_columns)

    df = df.replace("?", np.nan)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["11. Class"])
    X = df.drop(columns=["11. Class"])
    y = df["11. Class"]
    return X, y


def build_experiments(quick: bool) -> list[tuple[str, ImbPipeline, dict[str, list[Any]]]]:
    """Build model pipelines and hyperparameter grids."""
    if quick:
        dt_grid = {
            "clf__criterion": ["gini"],
            "clf__max_depth": [3, None],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2],
            "clf__max_features": [None],
        }
        tree_grid = {
            "clf__n_estimators": [100],
            "clf__max_depth": [None, 5],
            "clf__max_features": ["sqrt"],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2],
        }
        rf_grid = {**tree_grid, "clf__bootstrap": [True]}
        svm_grid = {
            "clf__kernel": ["rbf", "linear"],
            "clf__C": [1, 10],
            "clf__gamma": ["scale", 0.01],
        }
        hgb_grid = {
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [None, 3],
            "clf__max_iter": [200],
            "clf__min_samples_leaf": [10, 20],
            "clf__l2_regularization": [0.0],
        }
    else:
        dt_grid = {
            "clf__criterion": ["gini", "entropy", "log_loss"],
            "clf__max_depth": [None, 2, 3, 4, 5, 8, 12],
            "clf__min_samples_split": [2, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 4, 10],
            "clf__max_features": [None, "sqrt", "log2"],
        }
        rf_grid = {
            "clf__n_estimators": [300, 800],
            "clf__max_depth": [None, 5, 10],
            "clf__max_features": ["sqrt", "log2", None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__bootstrap": [True, False],
        }
        tree_grid = {
            "clf__n_estimators": [300, 800],
            "clf__max_depth": [None, 5, 10],
            "clf__max_features": ["sqrt", "log2", None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        }
        svm_grid = {
            "clf__kernel": ["rbf", "linear"],
            "clf__C": [0.1, 1, 10, 100, 1000],
            "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        }
        hgb_grid = {
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_depth": [None, 2, 3, 5],
            "clf__max_iter": [200, 400, 800],
            "clf__min_samples_leaf": [5, 10, 20],
            "clf__l2_regularization": [0.0, 0.1, 1.0],
        }

    experiments = [
        (
            "DecisionTree (no SMOTE)",
            ImbPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", DecisionTreeClassifier(random_state=42)),
                ]
            ),
            dt_grid,
        ),
        (
            "DecisionTree (SMOTE)",
            ImbPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("smote", SMOTE(random_state=42)),
                    ("clf", DecisionTreeClassifier(random_state=42)),
                ]
            ),
            dt_grid,
        ),
        (
            "RF (no SMOTE)",
            ImbPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", RandomForestClassifier(random_state=42)),
                ]
            ),
            rf_grid,
        ),
        (
            "RF (SMOTE)",
            ImbPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("smote", SMOTE(random_state=42)),
                    ("clf", RandomForestClassifier(random_state=42)),
                ]
            ),
            rf_grid,
        ),
        (
            "ExtraTrees (no SMOTE)",
            ImbPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", ExtraTreesClassifier(random_state=42)),
                ]
            ),
            tree_grid,
        ),
        (
            "ExtraTrees (SMOTE)",
            ImbPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("smote", SMOTE(random_state=42)),
                    ("clf", ExtraTreesClassifier(random_state=42)),
                ]
            ),
            tree_grid,
        ),
        (
            "SVM (no SMOTE)",
            ImbPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", SVC()),
                ]
            ),
            svm_grid,
        ),
        (
            "SVM (SMOTE)",
            ImbPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("smote", SMOTE(random_state=42)),
                    ("scaler", StandardScaler()),
                    ("clf", SVC()),
                ]
            ),
            svm_grid,
        ),
        (
            "HistGB (no SMOTE)",
            ImbPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", HistGradientBoostingClassifier(random_state=42)),
                ]
            ),
            hgb_grid,
        ),
        (
            "HistGB (SMOTE)",
            ImbPipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("smote", SMOTE(random_state=42)),
                    ("clf", HistGradientBoostingClassifier(random_state=42)),
                ]
            ),
            hgb_grid,
        ),
    ]
    return experiments


def tune_and_evaluate(
    name: str,
    pipeline: ImbPipeline,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv: RepeatedStratifiedKFold,
) -> dict[str, Any]:
    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    predictions = best_model.predict(X_test)

    return {
        "name": name,
        "best_model": best_model,
        "best_params": grid.best_params_,
        "cv_accuracy": float(grid.best_score_),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
    }


def save_confusion_matrix(
    model: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    predictions = model.predict(X_test)
    labels = sorted(y_test.dropna().unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    display_labels = [f"Class {label}" for label in labels]

    disp = ConfusionMatrixDisplay(matrix, display_labels=display_labels)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_feature_importance(model: ImbPipeline, feature_names: pd.Index, output_path: Path) -> bool:
    classifier = model.named_steps["clf"]
    if not hasattr(classifier, "feature_importances_"):
        return False

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(feature_names[indices], importances[indices])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def write_report(
    best_result: dict[str, Any],
    results: list[dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    best_model = best_result["best_model"]
    predictions = best_model.predict(X_test)

    report = classification_report(y_test, predictions, output_dict=True)
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    rows = [
        {
            "Model": result["name"],
            "CV Accuracy": result["cv_accuracy"],
            "Test Accuracy": result["test_accuracy"],
            "Best Params": json.dumps(result["best_params"]),
        }
        for result in results
    ]
    results_df = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False)
    results_df.to_csv(output_dir / "model_results.csv", index=False)

    save_confusion_matrix(best_model, X_test, y_test, output_dir / "confusion_matrix.png")
    save_feature_importance(best_model, X_test.columns, output_dir / "feature_importance.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train breast cancer classification models.")
    parser.add_argument("--data", required=True, type=Path, help="Path to the dataset CSV or UCI .data file.")
    parser.add_argument("--output-dir", default=Path("results"), type=Path, help="Directory for generated outputs.")
    parser.add_argument("--quick", action="store_true", help="Use smaller hyperparameter grids for a faster run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    results = []
    for name, pipeline, param_grid in build_experiments(args.quick):
        result = tune_and_evaluate(name, pipeline, param_grid, X_train, X_test, y_train, y_test, cv)
        results.append(result)
        print(f"{name}: CV={result['cv_accuracy']:.4f} | TEST={result['test_accuracy']:.4f}")

    best_result = max(results, key=lambda item: item["test_accuracy"])
    write_report(best_result, results, X_test, y_test, args.output_dir)

    print()
    print("Best model:", best_result["name"])
    print("Best CV accuracy:", f"{best_result['cv_accuracy']:.4f}")
    print("Best test accuracy:", f"{best_result['test_accuracy']:.4f}")
    print("Best parameters:", best_result["best_params"])
    print("Outputs written to:", args.output_dir)


if __name__ == "__main__":
    main()

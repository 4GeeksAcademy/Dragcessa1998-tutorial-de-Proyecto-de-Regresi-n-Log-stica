from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "bank-marketing-campaign-data.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
TARGET_COLUMN = "y"
LEAKAGE_COLUMNS = ["duration"]


def resolve_raw_data_path(default_path: Path = RAW_DATA_PATH) -> Path:
    if default_path.exists():
        return default_path

    csv_files = sorted(default_path.parent.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    raise FileNotFoundError(
        "No se encontro el archivo bank-marketing-campaign-data.csv en data/raw/."
    )


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    dataset_path = resolve_raw_data_path(path or RAW_DATA_PATH)
    return pd.read_csv(dataset_path, sep=";")


def clean_bank_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.replace("unknown", np.nan)
    data = data.drop_duplicates().reset_index(drop=True)
    return data


def prepare_modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=LEAKAGE_COLUMNS, errors="ignore").copy()


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_feature_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"no": 0, "yes": 1})
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_model_pipeline(
    X: pd.DataFrame,
    C: float = 1.0,
    class_weight: Optional[str] = None,
    max_iter: int = 2000,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            (
                "model",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }


def optimize_model(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    search = GridSearchCV(
        estimator=build_model_pipeline(X_train, max_iter=3000),
        param_grid={
            "model__C": [0.01, 0.1, 1.0, 3.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        scoring="f1",
        cv=5,
    )
    search.fit(X_train, y_train)
    return search


def save_processed_data(
    clean_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_dir / "bank_marketing_clean.csv", index=False)
    train_df.to_csv(output_dir / "bank_marketing_train.csv", index=False)
    test_df.to_csv(output_dir / "bank_marketing_test.csv", index=False)
    metrics_df.to_csv(output_dir / "bank_marketing_model_comparison.csv", index=False)


def main() -> None:
    raw_df = load_dataset()
    clean_df = clean_bank_data(raw_df)
    modeling_df = prepare_modeling_data(clean_df)
    train_df, test_df = split_data(modeling_df)

    X_train, y_train = prepare_feature_target(train_df)
    X_test, y_test = prepare_feature_target(test_df)

    baseline_model = build_model_pipeline(X_train)
    baseline_model.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test)

    search = optimize_model(X_train, y_train)
    optimized_metrics = evaluate_model(search.best_estimator_, X_test, y_test)

    comparison_df = pd.DataFrame(
        [
            {"model": "baseline_logistic_regression", **baseline_metrics},
            {"model": "optimized_logistic_regression", **optimized_metrics},
        ]
    )

    save_processed_data(modeling_df, train_df, test_df, comparison_df)

    print(f"Archivo cargado: {resolve_raw_data_path()}")
    print(f"Registros originales: {len(raw_df):,}")
    print(f"Registros limpios: {len(clean_df):,}")
    print(f"Variables usadas para modelar: {modeling_df.shape[1] - 1}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print("\nMetricas baseline:")
    print(baseline_metrics)
    print("\nMejores hiperparametros:")
    print(search.best_params_)
    print("\nMetricas optimizadas:")
    print(optimized_metrics)
    print(f"\nArchivos guardados en: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "AB_NYC_2019.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
DROP_COLUMNS = ["id", "name", "host_id", "host_name", "last_review"]


def resolve_raw_data_path(default_path: Path = RAW_DATA_PATH) -> Path:
    """Find the expected CSV or fall back to the only CSV present in data/raw."""
    if default_path.exists():
        return default_path

    csv_files = sorted(default_path.parent.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    raise FileNotFoundError(
        "No se encontro el archivo de datos en data/raw. "
        "Descarga AB_NYC_2019.csv y guardalo en data/raw/."
    )


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    dataset_path = resolve_raw_data_path(path or RAW_DATA_PATH)
    return pd.read_csv(dataset_path)


def clean_airbnb_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.drop_duplicates()

    data["last_review"] = pd.to_datetime(data["last_review"], errors="coerce")
    data["reviews_per_month"] = data["reviews_per_month"].fillna(0)
    data["has_reviews"] = (data["number_of_reviews"] > 0).astype(int)

    if data["last_review"].notna().any():
        reference_date = data["last_review"].max()
        data["days_since_last_review"] = (reference_date - data["last_review"]).dt.days
        max_gap = int(data["days_since_last_review"].dropna().max())
        data["days_since_last_review"] = data["days_since_last_review"].fillna(max_gap)
    else:
        data["days_since_last_review"] = 0

    data = data.drop(columns=DROP_COLUMNS, errors="ignore")
    return data


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stratify_values = None
    if "room_type" in df.columns and df["room_type"].nunique() > 1:
        stratify_values = df["room_type"]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_processed_data(
    clean_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_dir / "airbnb_nyc_clean.csv", index=False)
    train_df.to_csv(output_dir / "airbnb_nyc_train.csv", index=False)
    test_df.to_csv(output_dir / "airbnb_nyc_test.csv", index=False)


def main() -> None:
    raw_df = load_dataset()
    clean_df = clean_airbnb_data(raw_df)
    train_df, test_df = split_data(clean_df)
    save_processed_data(clean_df, train_df, test_df)

    print(f"Archivo cargado: {resolve_raw_data_path()}")
    print(f"Registros originales: {len(raw_df):,}")
    print(f"Registros procesados: {len(clean_df):,}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Datos guardados en: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
]


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load the bike-sharing dataset from a CSV file.

    Args:
        file_path:
            Path to the CSV dataset.

    Returns:
        A pandas DataFrame containing the raw dataset.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    return pd.read_csv(file_path)


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the basic structure and quality of the dataset.

    Checks:
        - Expected columns are present.
        - There are no missing values.
        - There are no duplicate rows.
        - The target definition is consistent.

    Args:
        df:
            Dataset to validate.

    Raises:
        ValueError:
            If any validation check fails.
    """
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    missing_values = df.isna().sum().sum()

    if missing_values > 0:
        raise ValueError(
            f"Dataset contains {missing_values} missing values."
        )

    duplicate_rows = df.duplicated().sum()

    if duplicate_rows > 0:
        raise ValueError(
            f"Dataset contains {duplicate_rows} duplicate rows."
        )

    if not (df["cnt"] == df["casual"] + df["registered"]).all():
        raise ValueError(
            "Target validation failed: "
            "cnt must equal casual + registered."
        )


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data preparation.

    Operations:
        - Convert dteday to datetime.
        - Return a copy of the DataFrame.

    Feature engineering is intentionally handled separately.

    Args:
        df:
            Raw bike-sharing dataset.

    Returns:
        Prepared DataFrame.
    """
    prepared_df = df.copy()

    prepared_df["dteday"] = pd.to_datetime(
        prepared_df["dteday"]
    )

    return prepared_df


def load_and_prepare_data(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Load, validate, and prepare the bike-sharing dataset.

    Args:
        file_path:
            Path to the CSV dataset.

    Returns:
        Validated and prepared DataFrame.
    """
    df = load_data(file_path)

    validate_data(df)

    return prepare_data(df)
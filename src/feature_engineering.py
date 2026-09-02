from __future__ import annotations

import pandas as pd


BASE_FEATURES = [
    "season",
    "holiday",
    "workingday",
    "weathersit",
    "temp",
    "hum",
    "windspeed",
]

ENGINEERED_FEATURES = [
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "is_weekend",
    "is_rush_hour",
]

TARGET = "cnt"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the date and hour columns.

    Features created:
        - year
        - month
        - day
        - hour
        - weekday
        - is_weekend
        - is_rush_hour

    Args:
        df:
            Prepared bike-sharing dataset.

    Returns:
        DataFrame with engineered time features.
    """
    result = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(result["dteday"]):
        result["dteday"] = pd.to_datetime(result["dteday"])

    result["year"] = result["dteday"].dt.year
    result["month"] = result["dteday"].dt.month
    result["day"] = result["dteday"].dt.day

    result["hour"] = result["hr"]
    result["weekday"] = result["weekday"]

    result["is_weekend"] = (
        result["weekday"].isin([0, 6]).astype(int)
    )

    result["is_rush_hour"] = (
        result["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    )

    return result


def create_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create the final feature matrix and target variable.

    Excluded columns:
        - instant: record identifier
        - casual: component of the target
        - registered: component of the target

    Target:
        - cnt

    Args:
        df:
            Prepared bike-sharing dataset.

    Returns:
        A tuple containing:
            X: Feature matrix
            y: Target variable
    """
    result = add_time_features(df)

    feature_columns = (
        BASE_FEATURES
        + ENGINEERED_FEATURES
    )

    X = result[feature_columns].copy()
    y = result[TARGET].copy()

    return X, y
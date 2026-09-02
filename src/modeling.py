from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


CATEGORICAL_FEATURES = [
    "season",
    "holiday",
    "workingday",
    "weathersit",
    "year",
    "month",
    "hour",
    "weekday",
    "is_weekend",
    "is_rush_hour",
]

NUMERICAL_FEATURES = [
    "temp",
    "hum",
    "windspeed",
    "day",
]


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Split the dataset chronologically into training and test sets.

    The data are not shuffled because the project uses a time-based
    evaluation strategy.

    Args:
        X:
            Feature matrix.
        y:
            Target variable.
        test_size:
            Fraction of observations reserved for testing.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return X_train, X_test, y_train, y_test


def build_linear_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline for Linear Regression.

    Numerical features are standardized.
    Categorical features are one-hot encoded.
    """
    numerical_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_tree_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline for tree-based models.

    Numerical features are passed through without scaling.
    Categorical features are one-hot encoded.
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    """
    Build the machine learning models used in the project.

    Returns:
        Dictionary containing model pipelines.
    """
    linear_regression = Pipeline(
        steps=[
            (
                "preprocessor",
                build_linear_preprocessor(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    decision_tree = Pipeline(
        steps=[
            (
                "preprocessor",
                build_tree_preprocessor(),
            ),
            (
                "model",
                DecisionTreeRegressor(
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            (
                "preprocessor",
                build_tree_preprocessor(),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "Linear Regression": linear_regression,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
    }


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate regression evaluation metrics.

    Metrics:
        - MAE
        - RMSE
        - R²
    """
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def evaluate_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Evaluate a simple mean-based baseline.

    The prediction for every test observation is the mean
    target value from the training set.
    """
    baseline_value = y_train.mean()

    predictions = np.full(
        shape=len(y_test),
        fill_value=baseline_value,
    )

    return calculate_metrics(
        y_test,
        predictions,
    )


def train_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    """
    Train all machine learning models.

    Returns:
        Dictionary containing fitted model pipelines.
    """
    trained_models = {}

    for name, model in models.items():
        model.fit(
            X_train,
            y_train,
        )

        trained_models[name] = model

    return trained_models


def evaluate_models(
    models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate fitted models on the test set.

    Returns:
        DataFrame containing MAE, RMSE, and R² for each model.
    """
    results = []

    for name, model in models.items():
        predictions = model.predict(X_test)

        metrics = calculate_metrics(
            y_test,
            predictions,
        )

        results.append(
            {
                "Model": name,
                **metrics,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )


def get_predictions(
    model: Pipeline,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions from a fitted model.
    """
    return model.predict(X_test)


def get_feature_importance(
    model: Pipeline,
) -> pd.DataFrame:
    """
    Extract feature importance from a fitted tree-based model.

    Returns:
        DataFrame containing feature names and importance values.
    """
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        raise ValueError(
            "Feature importance is only available "
            "for tree-based models."
        )

    feature_names = preprocessor.get_feature_names_out()
    importances = estimator.feature_importances_

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    )

    return (
        importance_df
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )
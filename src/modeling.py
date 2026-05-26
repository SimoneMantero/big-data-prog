from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "year",
    "unemployment_rate_lag1",
    "employed_thousands_lag1",
    "log_gdp_lag1",
    "gdp_per_employed_lag1",
    "neet_rate_lag1",
]
CATEGORICAL_FEATURES = ["REF_AREA"]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


@dataclass(frozen=True)
class ForecastSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def chronological_split(sample: pd.DataFrame, train_end_year: int = 2023) -> ForecastSplit:
    """Split the panel chronologically to avoid training on future observations."""
    train = sample.loc[sample["year"] <= train_end_year].copy()
    test = sample.loc[sample["year"] > train_end_year].copy()
    return ForecastSplit(train=train, test=test)


def regression_formula(target: str) -> str:
    """Statsmodels formula for a classical OLS panel-style regression."""
    terms = " + ".join(NUMERIC_FEATURES + ["C(REF_AREA)"])
    return f"{target} ~ {terms}"


def make_ml_pipeline(model) -> Pipeline:
    """Build a preprocessing + ML model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def prediction_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return compact regression metrics."""
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((np.asarray(y_true) - y_pred) / np.asarray(y_true))) * 100
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_%": mape,
        "R2": r2_score(y_true, y_pred),
    }


def fit_forecast_models(
    sample: pd.DataFrame,
    target: str,
    train_end_year: int = 2023,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit naive, OLS and ML models, returning metrics and row-level predictions."""
    split = chronological_split(sample, train_end_year=train_end_year)
    train, test = split.train, split.test

    if train.empty or test.empty:
        raise ValueError("Train and test split must both contain observations.")

    predictions = []

    lag_col = "unemployment_rate_lag1" if target == "unemployment_rate" else "employed_thousands_lag1"
    predictions.append(
        pd.DataFrame(
            {
                "Territorio": test["Territorio"].values,
                "REF_AREA": test["REF_AREA"].values,
                "year": test["year"].values,
                "target": target,
                "model": "Naive lag-1",
                "actual": test[target].values,
                "prediction": test[lag_col].values,
            }
        )
    )

    ols = smf.ols(regression_formula(target), data=train).fit()
    predictions.append(
        pd.DataFrame(
            {
                "Territorio": test["Territorio"].values,
                "REF_AREA": test["REF_AREA"].values,
                "year": test["year"].values,
                "target": target,
                "model": "OLS classica",
                "actual": test[target].values,
                "prediction": ols.predict(test).values,
            }
        )
    )

    ml_models = {
        "Ridge ML lineare": Ridge(alpha=1.0),
        "ElasticNet ML lineare": ElasticNet(alpha=0.02, l1_ratio=0.3, max_iter=10000, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=2,
            random_state=random_state,
        ),
    }

    for name, model in ml_models.items():
        pipeline = make_ml_pipeline(model)
        pipeline.fit(train[FEATURES], train[target])
        predictions.append(
            pd.DataFrame(
                {
                    "Territorio": test["Territorio"].values,
                    "REF_AREA": test["REF_AREA"].values,
                    "year": test["year"].values,
                    "target": target,
                    "model": name,
                    "actual": test[target].values,
                    "prediction": pipeline.predict(test[FEATURES]),
                }
            )
        )

    prediction_df = pd.concat(predictions, ignore_index=True)
    metrics = []
    for model_name, group in prediction_df.groupby("model"):
        row = {"target": target, "model": model_name, "n_test": len(group)}
        row.update(prediction_metrics(group["actual"], group["prediction"]))
        metrics.append(row)

    metrics_df = pd.DataFrame(metrics).sort_values(["target", "MAE"]).reset_index(drop=True)
    return metrics_df, prediction_df


def run_all_forecasts(
    samples: dict[str, pd.DataFrame],
    train_end_year: int = 2023,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run forecasting comparison for multiple targets."""
    all_metrics = []
    all_predictions = []
    for target, sample in samples.items():
        metrics, predictions = fit_forecast_models(sample, target, train_end_year=train_end_year)
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    return (
        pd.concat(all_metrics, ignore_index=True),
        pd.concat(all_predictions, ignore_index=True),
    )


def rolling_backtest(
    samples: dict[str, pd.DataFrame],
    train_end_years: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window backtests across multiple train/test years."""
    if train_end_years is None:
        train_end_years = [2021, 2022, 2023]

    all_metrics = []
    all_predictions = []
    for train_end_year in train_end_years:
        metrics, predictions = run_all_forecasts(samples, train_end_year=train_end_year)
        metrics.insert(0, "train_end_year", train_end_year)
        predictions.insert(0, "train_end_year", train_end_year)
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    return (
        pd.concat(all_metrics, ignore_index=True),
        pd.concat(all_predictions, ignore_index=True),
    )


def summarize_backtest_metrics(backtest_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rolling-backtest metrics by target and model."""
    return (
        backtest_metrics.groupby(["target", "model"], as_index=False)
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            MAPE_mean=("MAPE_%", "mean"),
            R2_mean=("R2", "mean"),
            windows=("train_end_year", "nunique"),
        )
        .sort_values(["target", "MAE_mean"])
        .reset_index(drop=True)
    )


def fitted_random_forest_importance(
    sample: pd.DataFrame,
    target: str = "unemployment_rate",
    train_end_year: int = 2023,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit a Random Forest and return grouped feature importances."""
    split = chronological_split(sample, train_end_year=train_end_year)
    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        random_state=random_state,
    )
    pipeline = make_ml_pipeline(model)
    pipeline.fit(split.train[FEATURES], split.train[target])

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    raw = pd.DataFrame({"feature": feature_names, "importance": importances})
    raw["feature_group"] = (
        raw["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__REF_AREA_", "territory_", regex=False)
    )
    raw.loc[raw["feature_group"].str.startswith("territory_"), "feature_group"] = "territory_dummies"
    return (
        raw.groupby("feature_group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def build_critical_sample(sample: pd.DataFrame, target: str = "unemployment_rate") -> pd.DataFrame:
    """Create a binary target: 1 if a region is above the yearly median unemployment rate."""
    out = sample.copy()
    yearly_median = out.groupby("year")[target].transform("median")
    out["critical_region"] = (out[target] > yearly_median).astype(int)
    return out


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Return compact binary-classification metrics."""
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        out["ROC_AUC"] = roc_auc_score(y_true, y_proba)
    else:
        out["ROC_AUC"] = np.nan
    return out


def fit_critical_classifiers(
    sample: pd.DataFrame,
    train_end_year: int = 2023,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare classifiers for predicting high-unemployment critical regions."""
    critical_sample = build_critical_sample(sample)
    split = chronological_split(critical_sample, train_end_year=train_end_year)
    train, test = split.train, split.test

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        ),
    }

    predictions = []
    metrics = []
    for name, clf in classifiers.items():
        pipeline = make_ml_pipeline(clf)
        pipeline.fit(train[FEATURES], train["critical_region"])
        pred = pipeline.predict(test[FEATURES])
        proba = pipeline.predict_proba(test[FEATURES])[:, 1]
        predictions.append(
            pd.DataFrame(
                {
                    "Territorio": test["Territorio"].values,
                    "REF_AREA": test["REF_AREA"].values,
                    "year": test["year"].values,
                    "model": name,
                    "actual_critical": test["critical_region"].values,
                    "predicted_critical": pred,
                    "critical_probability": proba,
                    "actual_unemployment_rate": test["unemployment_rate"].values,
                }
            )
        )
        row = {"model": name, "n_test": len(test)}
        row.update(classification_metrics(test["critical_region"], pred, proba))
        metrics.append(row)

    return (
        pd.DataFrame(metrics).sort_values("F1", ascending=False).reset_index(drop=True),
        pd.concat(predictions, ignore_index=True),
    )

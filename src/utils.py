from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


RAW_FILES = {
    "unemployment": DATA_RAW / "disoccupazione_regionale_eta.csv",
    "employment": DATA_RAW / "occupati_regionale_eta.csv",
    "gdp": DATA_RAW / "pil_lato_produzione.csv",
    "neet_rate": DATA_RAW / "neet_incidenza_regionale.csv",
    "neet_count": DATA_RAW / "neet_valori_regionale.csv",
}


# Italian regions plus autonomous provinces, excluding Italy and macro-areas.
REGION_CODES = [
    "ITC1",
    "ITC2",
    "ITC3",
    "ITC4",
    "ITD1",
    "ITD2",
    "ITD3",
    "ITD4",
    "ITD5",
    "ITE1",
    "ITE2",
    "ITE3",
    "ITE4",
    "ITF1",
    "ITF2",
    "ITF3",
    "ITF4",
    "ITF5",
    "ITF6",
    "ITG1",
    "ITG2",
]


REGION_NAME_FIXES = {
    "'Valle d\"'Aosta / Vallée d\"'Aoste'": "Valle d'Aosta",
    "Provincia Autonoma Bolzano / Bozen": "Bolzano",
    "Provincia Autonoma Trento": "Trento",
}


def read_istat_csv(path: str | Path) -> pd.DataFrame:
    """Read an ISTAT CSV and normalize the most frequently used columns."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(col).strip() for col in df.columns]

    if "TIME_PERIOD" in df.columns:
        df["TIME_PERIOD"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce")
        if df["TIME_PERIOD"].notna().all():
            df["TIME_PERIOD"] = df["TIME_PERIOD"].astype(int)
        else:
            df["TIME_PERIOD"] = df["TIME_PERIOD"].astype("Int64")

    if "Osservazione" in df.columns:
        df["Osservazione"] = pd.to_numeric(df["Osservazione"], errors="coerce")

    if "Territorio" in df.columns:
        df["Territorio"] = df["Territorio"].replace(REGION_NAME_FIXES)

    return df


def filter_regions(df: pd.DataFrame, region_codes: Iterable[str] = REGION_CODES) -> pd.DataFrame:
    """Keep only regional observations used for the modelling panel."""
    codes = set(region_codes)
    return df.loc[df["REF_AREA"].isin(codes)].copy()


def dataset_overview() -> pd.DataFrame:
    """Return a compact overview of raw dataset dimensions and coverage."""
    rows = []
    for name, path in RAW_FILES.items():
        df = read_istat_csv(path)
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "columns": len(df.columns),
                "year_min": int(df["TIME_PERIOD"].min()) if "TIME_PERIOD" in df else np.nan,
                "year_max": int(df["TIME_PERIOD"].max()) if "TIME_PERIOD" in df else np.nan,
                "territories": df["REF_AREA"].nunique() if "REF_AREA" in df else np.nan,
                "missing_obs": int(df["Osservazione"].isna().sum()) if "Osservazione" in df else np.nan,
            }
        )
    return pd.DataFrame(rows)


def regional_coverage_overview() -> pd.DataFrame:
    """Return coverage after keeping only the regional modelling geography."""
    specs = [
        ("unemployment", RAW_FILES["unemployment"]),
        ("employment", RAW_FILES["employment"]),
        ("gdp", RAW_FILES["gdp"]),
        ("neet_rate", RAW_FILES["neet_rate"]),
        ("neet_count", RAW_FILES["neet_count"]),
    ]
    rows = []
    for name, path in specs:
        df = filter_regions(read_istat_csv(path))
        rows.append(
            {
                "dataset": name,
                "rows_regional": len(df),
                "territories": df["REF_AREA"].nunique(),
                "year_min": int(df["TIME_PERIOD"].min()),
                "year_max": int(df["TIME_PERIOD"].max()),
                "years": df["TIME_PERIOD"].nunique(),
                "missing_obs": int(df["Osservazione"].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def load_unemployment(age: str = "15-64 anni", sex: str = "Totale") -> pd.DataFrame:
    """Load regional unemployment rate for a chosen age and sex class."""
    df = filter_regions(read_istat_csv(RAW_FILES["unemployment"]))
    df = df.loc[(df["Età"] == age) & (df["Sesso"] == sex)].copy()
    return (
        df[["REF_AREA", "Territorio", "TIME_PERIOD", "Osservazione"]]
        .rename(columns={"TIME_PERIOD": "year", "Osservazione": "unemployment_rate"})
        .sort_values(["REF_AREA", "year"])
        .reset_index(drop=True)
    )


def load_employment(age: str = "15-64 anni", sex: str = "Totale") -> pd.DataFrame:
    """Load regional employed people, expressed in thousands, for age and sex class."""
    df = filter_regions(read_istat_csv(RAW_FILES["employment"]))
    df = df.loc[(df["Età"] == age) & (df["Sesso"] == sex)].copy()
    return (
        df[["REF_AREA", "Territorio", "TIME_PERIOD", "Osservazione"]]
        .rename(columns={"TIME_PERIOD": "year", "Osservazione": "employed_thousands"})
        .sort_values(["REF_AREA", "year"])
        .reset_index(drop=True)
    )


def load_gdp() -> pd.DataFrame:
    """Load regional GDP at current prices, expressed in millions of euros."""
    df = filter_regions(read_istat_csv(RAW_FILES["gdp"]))
    df = df.loc[df["Aggregato"] == "Prodotto interno lordo ai prezzi di mercato"].copy()
    gdp = (
        df[["REF_AREA", "Territorio", "TIME_PERIOD", "Osservazione"]]
        .rename(columns={"TIME_PERIOD": "year", "Osservazione": "gdp_million_eur"})
        .sort_values(["REF_AREA", "year"])
        .reset_index(drop=True)
    )
    gdp["gdp_yoy_pct"] = gdp.groupby("REF_AREA")["gdp_million_eur"].pct_change() * 100
    gdp["gdp_lag1"] = gdp.groupby("REF_AREA")["gdp_million_eur"].shift(1)
    return gdp


def load_neet_rate(age: str = "15-29 anni", sex: str = "Totale") -> pd.DataFrame:
    """Load regional NEET incidence rate, useful as a contextual labour-market feature."""
    df = filter_regions(read_istat_csv(RAW_FILES["neet_rate"]))
    df = df.loc[(df["Età"] == age) & (df["Sesso"] == sex)].copy()
    return (
        df[["REF_AREA", "Territorio", "TIME_PERIOD", "Osservazione"]]
        .rename(columns={"TIME_PERIOD": "year", "Osservazione": "neet_rate"})
        .sort_values(["REF_AREA", "year"])
        .reset_index(drop=True)
    )


def build_labour_panel(include_neet: bool = True) -> pd.DataFrame:
    """Merge unemployment, employment, GDP and optional NEET data by region and year."""
    panel = load_unemployment().merge(
        load_employment(),
        on=["REF_AREA", "Territorio", "year"],
        how="outer",
    )
    panel = panel.merge(load_gdp(), on=["REF_AREA", "Territorio", "year"], how="left")

    if include_neet:
        panel = panel.merge(load_neet_rate(), on=["REF_AREA", "Territorio", "year"], how="left")

    panel = panel.sort_values(["REF_AREA", "year"]).reset_index(drop=True)
    panel["log_gdp"] = np.log(panel["gdp_million_eur"])
    panel["gdp_per_employed_k_eur"] = panel["gdp_million_eur"] / panel["employed_thousands"]
    panel["unemployment_rate_lag1"] = panel.groupby("REF_AREA")["unemployment_rate"].shift(1)
    panel["employed_thousands_lag1"] = panel.groupby("REF_AREA")["employed_thousands"].shift(1)
    panel["gdp_lag1"] = panel.groupby("REF_AREA")["gdp_million_eur"].shift(1)
    panel["log_gdp_lag1"] = np.log(panel["gdp_lag1"])
    panel["gdp_per_employed_lag1"] = panel["gdp_lag1"] / panel["employed_thousands_lag1"]
    if "neet_rate" in panel.columns:
        panel["neet_rate_lag1"] = panel.groupby("REF_AREA")["neet_rate"].shift(1)
    return panel


def modelling_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with the core variables available for later supervised models."""
    core_cols = [
        "unemployment_rate",
        "employed_thousands",
        "gdp_million_eur",
        "gdp_yoy_pct",
        "log_gdp",
        "gdp_per_employed_k_eur",
    ]
    return df.dropna(subset=core_cols).copy()


FORECAST_FEATURES = [
    "year",
    "unemployment_rate_lag1",
    "employed_thousands_lag1",
    "gdp_lag1",
    "log_gdp_lag1",
    "gdp_per_employed_lag1",
    "neet_rate_lag1",
    "REF_AREA",
]


def build_forecasting_sample(target: str) -> pd.DataFrame:
    """Build a supervised sample using only previous-year information as features."""
    if target not in {"unemployment_rate", "employed_thousands"}:
        raise ValueError("target must be 'unemployment_rate' or 'employed_thousands'")

    panel = build_labour_panel(include_neet=True)
    required = [target, *FORECAST_FEATURES]
    sample = panel.dropna(subset=required).copy()
    return sample[["Territorio", target, *FORECAST_FEATURES]].reset_index(drop=True)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values as counts and percentages."""
    out = pd.DataFrame(
        {
            "missing": df.isna().sum(),
            "missing_pct": df.isna().mean() * 100,
        }
    )
    return out.loc[out["missing"] > 0].sort_values("missing_pct", ascending=False)


def save_processed_panel(path: str | Path | None = None) -> Path:
    """Build and save the merged panel for later modelling steps."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = Path(path) if path is not None else DATA_PROCESSED / "labour_panel.csv"
    build_labour_panel().to_csv(output_path, index=False)
    return output_path

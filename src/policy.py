from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_unemployed_thousands(employed_thousands: pd.Series, unemployment_rate: pd.Series) -> pd.Series:
    """Approximate unemployed people in thousands from employed stock and unemployment rate."""
    rate = unemployment_rate / 100
    return employed_thousands * rate / (1 - rate)


def percentile_rank(series: pd.Series, higher_is_risk: bool = True) -> pd.Series:
    """Return percentile ranks where larger values mean higher policy risk."""
    values = series if higher_is_risk else -series
    return values.rank(pct=True)


def build_policy_table(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    year: int = 2025,
    unemployment_model: str = "Random Forest",
) -> pd.DataFrame:
    """Combine forecast outputs with economic context to prioritize regions."""
    unemployment_pred = predictions.query(
        "target == 'unemployment_rate' and model == @unemployment_model and year == @year"
    )[["REF_AREA", "Territorio", "prediction"]].rename(columns={"prediction": "pred_unemployment_rate"})

    current = panel.loc[panel["year"] == year].copy()
    table = current.merge(unemployment_pred, on=["REF_AREA", "Territorio"], how="left")

    table["pred_unemployed_thousands"] = estimate_unemployed_thousands(
        table["employed_thousands_lag1"],
        table["pred_unemployment_rate"],
    )
    table["actual_unemployed_thousands"] = estimate_unemployed_thousands(
        table["employed_thousands"],
        table["unemployment_rate"],
    )

    table["risk_score"] = (
        0.35 * percentile_rank(table["pred_unemployment_rate"], higher_is_risk=True)
        + 0.25 * percentile_rank(table["neet_rate_lag1"], higher_is_risk=True)
        + 0.20 * percentile_rank(table["gdp_per_employed_lag1"], higher_is_risk=False)
        + 0.20 * percentile_rank(table["pred_unemployed_thousands"], higher_is_risk=True)
    )

    low_gdp_threshold = table["gdp_per_employed_lag1"].median()
    high_unemployment_threshold = table["pred_unemployment_rate"].median()
    high_neet_threshold = table["neet_rate_lag1"].median()

    conditions = [
        (table["pred_unemployment_rate"] >= high_unemployment_threshold)
        & (table["gdp_per_employed_lag1"] < low_gdp_threshold),
        (table["pred_unemployment_rate"] >= high_unemployment_threshold)
        & (table["gdp_per_employed_lag1"] >= low_gdp_threshold),
        (table["pred_unemployment_rate"] < high_unemployment_threshold)
        & (table["gdp_per_employed_lag1"] < low_gdp_threshold),
    ]
    choices = [
        "Priorita alta: disoccupazione alta e PIL per occupato basso",
        "Mismatch: disoccupazione alta ma base economica piu forte",
        "Produttivita fragile: bassa disoccupazione ma PIL per occupato basso",
    ]
    table["policy_cluster"] = np.select(conditions, choices, default="Tenuta relativa")
    table["neet_flag"] = np.where(table["neet_rate_lag1"] >= high_neet_threshold, "NEET alto", "NEET sotto mediana")

    columns = [
        "REF_AREA",
        "Territorio",
        "year",
        "pred_unemployment_rate",
        "unemployment_rate",
        "pred_unemployed_thousands",
        "actual_unemployed_thousands",
        "employed_thousands",
        "gdp_lag1",
        "gdp_per_employed_lag1",
        "neet_rate_lag1",
        "risk_score",
        "policy_cluster",
        "neet_flag",
    ]
    return table[columns].sort_values("risk_score", ascending=False).reset_index(drop=True)


def recommend_action(row: pd.Series) -> str:
    """Translate a region profile into a concise policy action."""
    cluster = row["policy_cluster"]
    neet_high = row["neet_flag"] == "NEET alto"

    if cluster.startswith("Priorita alta"):
        base = (
            "Intervento integrato: incentivi a investimenti produttivi locali, formazione tecnica "
            "mirata ai settori con domanda, rafforzamento dei centri per l'impiego e programmi "
            "di inserimento per giovani e disoccupati di lunga durata."
        )
    elif cluster.startswith("Mismatch"):
        base = (
            "Intervento di matching: orientamento, riqualificazione breve, accordi con imprese "
            "e universita/ITS per trasformare la base economica esistente in assorbimento occupazionale."
        )
    elif cluster.startswith("Produttivita fragile"):
        base = (
            "Intervento di produttivita: sostegno a innovazione, digitalizzazione e crescita dimensionale "
            "delle imprese per evitare occupazione fragile e basso valore aggiunto."
        )
    else:
        base = (
            "Monitoraggio: mantenere politiche attive leggere, osservare NEET e trend occupazionale, "
            "e intervenire se peggiorano gli indicatori anticipatori."
        )

    if neet_high:
        base += " Priorita aggiuntiva: ridurre NEET con apprendistato, ITS e percorsi scuola-lavoro regionali."
    return base

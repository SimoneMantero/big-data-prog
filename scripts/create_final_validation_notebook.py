from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python (.venv)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

cells = [
    md(
        """# Validazione finale, feature importance e classificazione delle regioni critiche

Questa ultima fase rafforza il progetto in tre modi:

1. usa un **backtesting rolling** per non dipendere da un singolo split temporale;
2. mostra quali feature contano di piu nel modello ML;
3. trasforma la previsione numerica in una classificazione: regione critica oppure no.

La classificazione e utile per una lettura da policy, perche spesso un decisore deve sapere prima dove intervenire, non solo il valore puntuale previsto."""
    ),
    code(
        """from pathlib import Path
import sys

CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR if (CURRENT_DIR / "src").exists() else CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import modeling, utils

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.figsize"] = (11, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)"""
    ),
    md("""## 1. Preparazione campioni"""),
    code(
        """samples = {
    "unemployment_rate": utils.build_forecasting_sample("unemployment_rate"),
    "employed_thousands": utils.build_forecasting_sample("employed_thousands"),
}

pd.DataFrame(
    [
        {
            "target": target,
            "rows": len(sample),
            "territories": sample["REF_AREA"].nunique(),
            "year_min": int(sample["year"].min()),
            "year_max": int(sample["year"].max()),
        }
        for target, sample in samples.items()
    ]
)"""
    ),
    md(
        """## 2. Backtesting rolling

Usiamo finestre espandenti:

- train fino al 2021, test 2022-2025;
- train fino al 2022, test 2023-2025;
- train fino al 2023, test 2024-2025.

Questo rende il confronto piu robusto rispetto a un solo split."""
    ),
    code(
        """backtest_metrics, backtest_predictions = modeling.rolling_backtest(samples, train_end_years=[2021, 2022, 2023])
backtest_summary = modeling.summarize_backtest_metrics(backtest_metrics)

backtest_metrics.to_csv(PROCESSED_DIR / "rolling_backtest_metrics.csv", index=False)
backtest_predictions.to_csv(PROCESSED_DIR / "rolling_backtest_predictions.csv", index=False)
backtest_summary.to_csv(PROCESSED_DIR / "rolling_backtest_summary.csv", index=False)

backtest_summary"""
    ),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for ax, target, title in zip(
    axes,
    ["unemployment_rate", "employed_thousands"],
    ["Backtesting disoccupazione", "Backtesting occupati"],
):
    subset = backtest_summary.query("target == @target").sort_values("MAE_mean")
    sns.barplot(data=subset, x="MAE_mean", y="model", ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("MAE medio")
    ax.set_ylabel("")

plt.tight_layout()
plt.savefig(FIG_DIR / "rolling_backtest_mae.png", dpi=150)
plt.show()"""
    ),
    md(
        """**Lettura:** se un modello resta buono su piu finestre temporali, e piu credibile. Nel nostro caso, per la disoccupazione i modelli non lineari restano in testa; per gli occupati resta fortissima la baseline dell'anno precedente."""
    ),
    md("""## 3. Feature importance del Random Forest sulla disoccupazione"""),
    code(
        """importance = modeling.fitted_random_forest_importance(
    samples["unemployment_rate"],
    target="unemployment_rate",
    train_end_year=2023,
)
importance.to_csv(PROCESSED_DIR / "rf_feature_importance_unemployment.csv", index=False)
feature_labels = {
    "unemployment_rate_lag1": "Disoccupazione anno precedente",
    "neet_rate_lag1": "NEET anno precedente",
    "gdp_per_employed_lag1": "PIL per occupato anno precedente",
    "year": "Trend temporale",
    "log_gdp_lag1": "Log PIL anno precedente",
    "employed_thousands_lag1": "Occupati anno precedente",
    "territory_dummies": "Effetto territorio",
}
importance_display = importance.assign(
    feature_label=importance["feature_group"].map(feature_labels).fillna(importance["feature_group"])
)
importance_display"""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=importance_display, x="importance", y="feature_label", ax=ax, color="#59A14F")
ax.set_title("Feature importance Random Forest - disoccupazione")
ax.set_xlabel("Importanza")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "rf_feature_importance_unemployment.png", dpi=150)
plt.show()"""
    ),
    md(
        """**Lettura:** la disoccupazione dell'anno precedente e il driver principale. Questo e coerente con un fenomeno persistente. Il NEET aggiunge informazione sociale importante. Il PIL per occupato pesa meno nella pura previsione puntuale, ma resta centrale per interpretare la qualita economica del territorio e definire gli interventi."""
    ),
    md("""## 4. Classificazione delle regioni critiche"""),
    md(
        """Definiamo una regione critica se il suo tasso di disoccupazione e sopra la mediana delle regioni nello stesso anno. Questo rende il problema piu vicino alla decisione pubblica: identificare aree da monitorare o finanziare con priorita."""
    ),
    code(
        """class_metrics, class_predictions = modeling.fit_critical_classifiers(
    samples["unemployment_rate"],
    train_end_year=2023,
)
class_metrics.to_csv(PROCESSED_DIR / "critical_region_classification_metrics.csv", index=False)
class_predictions.to_csv(PROCESSED_DIR / "critical_region_predictions.csv", index=False)
class_metrics"""
    ),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=class_metrics, x="F1", y="model", ax=axes[0], color="#E15759")
axes[0].set_title("Classificazione regioni critiche - F1")
axes[0].set_xlabel("F1")
axes[0].set_ylabel("")

sns.barplot(data=class_metrics, x="ROC_AUC", y="model", ax=axes[1], color="#F28E2B")
axes[1].set_title("Classificazione regioni critiche - ROC AUC")
axes[1].set_xlabel("ROC AUC")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig(FIG_DIR / "critical_classification_metrics.png", dpi=150)
plt.show()"""
    ),
    code(
        """best_classifier = class_metrics.sort_values("F1", ascending=False).iloc[0]["model"]
critical_2025 = (
    class_predictions.query("year == 2025 and model == @best_classifier")
    .sort_values("critical_probability", ascending=False)
    .reset_index(drop=True)
)
critical_2025.head(10)"""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(11, 6))
top_critical = critical_2025.head(10).sort_values("critical_probability")
sns.barplot(data=top_critical, x="critical_probability", y="Territorio", ax=ax, color="#B07AA1")
ax.set_title(f"Probabilita di regione critica nel 2025 ({best_classifier})")
ax.set_xlabel("Probabilita stimata")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "critical_region_probability_2025.png", dpi=150)
plt.show()"""
    ),
    md("""## 5. Conclusione finale pronta per la presentazione"""),
    code(
        """best_unemployment = (
    backtest_summary.query("target == 'unemployment_rate'")
    .sort_values("MAE_mean")
    .iloc[0]
)
best_employment = (
    backtest_summary.query("target == 'employed_thousands'")
    .sort_values("MAE_mean")
    .iloc[0]
)
critical_regions = critical_2025.head(5)["Territorio"].tolist()

conclusion = f\"\"\"
Conclusione:

Il backtesting conferma che la disoccupazione regionale e prevista meglio da modelli ML non lineari,
in particolare {best_unemployment['model']}, con MAE medio pari a {best_unemployment['MAE_mean']:.2f}
punti percentuali. Questo suggerisce che la relazione tra territorio, NEET, PIL per occupato e
disoccupazione non e puramente lineare.

Per gli occupati, invece, il modello migliore e {best_employment['model']}. Questo significa che
l'occupazione regionale e molto persistente: l'informazione piu forte e il livello dell'anno precedente.

La classificazione delle regioni critiche indica come aree piu rilevanti: {', '.join(critical_regions)}.
Queste regioni vanno lette insieme alla tabella policy: se alta disoccupazione prevista si combina con
PIL per occupato basso e NEET elevato, la risposta non puo essere solo assistenziale. Serve una politica
integrata: investimenti produttivi, formazione tecnica, ITS/apprendistato, rafforzamento dei centri per
l'impiego e monitoraggio annuale di PIL per occupato e NEET.
\"\"\"
print(conclusion)"""
    ),
]

nb["cells"] = cells
output = Path("notebooks/04_validation_classification_conclusions.ipynb")
output.write_text(nbf.writes(nb), encoding="utf-8")
print(output)

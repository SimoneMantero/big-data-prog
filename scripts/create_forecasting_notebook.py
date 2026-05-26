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
        """# Forecasting: OLS classica vs Machine Learning

In questa fase confrontiamo una regressione classica non-ML con modelli di Machine Learning per prevedere:

- `unemployment_rate`: tasso di disoccupazione regionale;
- `employed_thousands`: occupati regionali in migliaia.

L'obiettivo non e solo ottenere il numero migliore, ma costruire un confronto corretto e difendibile."""
    ),
    md(
        """## Strategia di forecasting

Per evitare leakage, le feature usate per prevedere l'anno `t` sono disponibili all'anno `t-1`:

- disoccupazione precedente;
- occupati precedenti;
- PIL precedente;
- log del PIL precedente;
- PIL per occupato precedente;
- NEET precedente;
- regione come variabile categorica.

Il PIL 2025 non e disponibile, quindi per prevedere il 2025 usiamo il PIL 2024. Il test e cronologico: training fino al 2023, valutazione su 2024 e 2025."""
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
import statsmodels.formula.api as smf

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
    md("""## 1. Dataset supervisionato per il forecast"""),
    code(
        """samples = {
    "unemployment_rate": utils.build_forecasting_sample("unemployment_rate"),
    "employed_thousands": utils.build_forecasting_sample("employed_thousands"),
}

sample_overview = pd.DataFrame(
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
)
sample_overview"""
    ),
    code(
        """samples["unemployment_rate"].head()"""
    ),
    md(
        """## 2. Modelli confrontati

**Baseline**
- `Naive lag-1`: predice che il valore dell'anno prossimo sara uguale a quello dell'anno precedente.

**Regressione normale senza ML**
- `OLS classica`: regressione lineare con dummy regionali tramite `statsmodels`.

**Machine Learning**
- `Ridge`: modello lineare ML regolarizzato.
- `ElasticNet`: modello lineare ML con regolarizzazione L1/L2.
- `Random Forest`: modello non lineare basato su molti alberi.
- `Gradient Boosting`: modello non lineare sequenziale basato su alberi deboli.

La baseline e fondamentale: se un modello ML non batte la previsione naive, non sta aggiungendo valore reale. Qui distinguiamo anche tra ML lineare e ML non lineare."""
    ),
    code(
        """train_end_year = 2023
metrics, predictions = modeling.run_all_forecasts(samples, train_end_year=train_end_year)

metrics_path = PROCESSED_DIR / "forecast_metrics.csv"
predictions_path = PROCESSED_DIR / "forecast_predictions.csv"
metrics.to_csv(metrics_path, index=False)
predictions.to_csv(predictions_path, index=False)

metrics"""
    ),
    md("""## 3. Confronto metriche"""),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for ax, target, title in zip(
    axes,
    ["unemployment_rate", "employed_thousands"],
    ["Disoccupazione: MAE in punti percentuali", "Occupati: MAE in migliaia"],
):
    subset = metrics.query("target == @target").sort_values("MAE")
    sns.barplot(data=subset, x="MAE", y="model", ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("MAE")
    ax.set_ylabel("")

plt.tight_layout()
plt.savefig(FIG_DIR / "forecast_mae_comparison.png", dpi=150)
plt.show()"""
    ),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for ax, target, title in zip(
    axes,
    ["unemployment_rate", "employed_thousands"],
    ["Disoccupazione: MAPE %", "Occupati: MAPE %"],
):
    subset = metrics.query("target == @target").sort_values("MAPE_%")
    sns.barplot(data=subset, x="MAPE_%", y="model", ax=ax, color="#59A14F")
    ax.set_title(title)
    ax.set_xlabel("MAPE %")
    ax.set_ylabel("")

plt.tight_layout()
plt.savefig(FIG_DIR / "forecast_mape_comparison.png", dpi=150)
plt.show()"""
    ),
    md(
        """## 4. Previsioni vs valori reali

I grafici seguenti mostrano il modello migliore per ogni target secondo MAE sul test 2024-2025."""
    ),
    code(
        """best_models = metrics.sort_values("MAE").groupby("target").first()["model"].to_dict()
best_models"""
    ),
    code(
        """best_predictions = pd.concat(
    [
        predictions.query("target == @target and model == @model_name")
        for target, model_name in best_models.items()
    ],
    ignore_index=True,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, target, title in zip(
    axes,
    ["unemployment_rate", "employed_thousands"],
    ["Disoccupazione", "Occupati"],
):
    subset = best_predictions.query("target == @target")
    sns.scatterplot(data=subset, x="actual", y="prediction", hue="year", palette="viridis", ax=ax)
    low = min(subset["actual"].min(), subset["prediction"].min())
    high = max(subset["actual"].max(), subset["prediction"].max())
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_title(f"{title}: reale vs previsto ({best_models[target]})")
    ax.set_xlabel("Valore reale")
    ax.set_ylabel("Valore previsto")

plt.tight_layout()
plt.savefig(FIG_DIR / "forecast_actual_vs_predicted.png", dpi=150)
plt.show()"""
    ),
    md("""## 5. Forecast 2025 e confronto con valori osservati"""),
    code(
        """forecast_2025 = predictions.query("year == 2025").copy()
forecast_2025["error"] = forecast_2025["prediction"] - forecast_2025["actual"]
forecast_2025["abs_error"] = forecast_2025["error"].abs()

forecast_2025.sort_values(["target", "model", "abs_error"]).head(12)"""
    ),
    code(
        """top_2025_errors = (
    forecast_2025.sort_values(["target", "model", "abs_error"], ascending=[True, True, False])
    .groupby(["target", "model"])
    .head(3)
)
top_2025_errors[["target", "model", "Territorio", "actual", "prediction", "error", "abs_error"]]"""
    ),
    md(
        """## 6. Regressione classica: coefficienti principali

Per la regressione OLS mostriamo i coefficienti delle variabili numeriche. Le dummy regionali sono incluse nel modello, ma non riportate qui per mantenere la tabella leggibile."""
    ),
    code(
        """ols_tables = []
for target, sample in samples.items():
    train = sample.query("year <= @train_end_year").copy()
    ols_model = smf.ols(modeling.regression_formula(target), data=train).fit()
    coefs = (
        ols_model.params.rename("coef")
        .to_frame()
        .join(ols_model.pvalues.rename("p_value"))
        .reset_index(names="feature")
    )
    coefs = coefs[coefs["feature"].isin(["Intercept", *modeling.NUMERIC_FEATURES])]
    coefs.insert(0, "target", target)
    ols_tables.append(coefs)

ols_coefficients = pd.concat(ols_tables, ignore_index=True)
ols_coefficients"""
    ),
    md(
        """## 7. Interpretazione provvisoria

- Per la disoccupazione i modelli ML tendono a migliorare sia OLS sia baseline naive: questo suggerisce una componente non lineare utile.
- Per gli occupati la baseline lag-1 e molto competitiva: l'occupazione regionale e altamente persistente, quindi un modello complesso deve battere una soglia molto alta.
- Il risultato non va letto come "ML sempre meglio": il confronto con baseline e OLS serve proprio a dimostrare quando il machine learning aggiunge valore e quando no.

Nella prossima fase possiamo rafforzare il progetto con backtesting rolling, feature di macro-area, feature di crescita e una sezione finale di classificazione delle regioni critiche."""
    ),
]

nb["cells"] = cells
output = Path("notebooks/02_forecasting_models.ipynb")
output.write_text(nbf.writes(nb), encoding="utf-8")
print(output)

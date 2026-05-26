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
        """# Domanda decisionale: dove intervenire contro la disoccupazione?

Dopo EDA e forecasting, trasformiamo gli output in una risposta concreta di policy.

**Domanda guida**

> Quali regioni italiane mostrano un rischio alto di disoccupazione nel 2025, considerando anche PIL, NEET e occupazione, e quale tipo di intervento pubblico/economico e piu coerente con il profilo del territorio?

Questa domanda collega direttamente il PIL ai predittori degli altri dataset: disoccupazione, occupati e NEET."""
    ),
    md(
        """## Logica dell'analisi

Usiamo il forecast 2025 della disoccupazione prodotto dai modelli precedenti. Per evitare di usare informazione futura, il rischio 2025 e costruito con variabili 2024:

- disoccupazione prevista 2025;
- occupati 2024, per stimare quanti disoccupati potrebbero esserci;
- PIL 2024;
- PIL per occupato 2024, come proxy di produttivita/forza economica del territorio;
- NEET 2024, come segnale di fragilita giovanile.

Il risultato e una graduatoria di priorita e una proposta di azione per cluster territoriale."""
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

from src import modeling, policy, utils

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.figsize"] = (11, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)"""
    ),
    md("""## 1. Ricalcolo dei modelli e scelta del modello per la disoccupazione"""),
    code(
        """samples = {
    "unemployment_rate": utils.build_forecasting_sample("unemployment_rate"),
    "employed_thousands": utils.build_forecasting_sample("employed_thousands"),
}
metrics, predictions = modeling.run_all_forecasts(samples, train_end_year=2023)
metrics"""
    ),
    code(
        """best_unemployment_model = (
    metrics.query("target == 'unemployment_rate'")
    .sort_values("MAE")
    .iloc[0]["model"]
)
best_unemployment_model"""
    ),
    md(
        """Nel nostro caso il miglior modello per la disoccupazione e non lineare. Questo e utile narrativamente: la relazione tra disoccupazione, NEET, PIL e territorio non sembra puramente lineare."""
    ),
    md("""## 2. Costruzione della tabella di priorita"""),
    code(
        """panel = utils.build_labour_panel(include_neet=True)
policy_table = policy.build_policy_table(
    panel,
    predictions,
    year=2025,
    unemployment_model=best_unemployment_model,
)

policy_path = PROCESSED_DIR / "policy_priority_table_2025.csv"
policy_table.to_csv(policy_path, index=False)

policy_table.head(10)"""
    ),
    md(
        """Lo score non e una previsione statistica autonoma: e un indice decisionale. Serve a mettere insieme rischio di disoccupazione, fragilita giovanile, massa potenziale di disoccupati e debolezza produttiva."""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(11, 7))
top = policy_table.head(10).sort_values("risk_score")
sns.barplot(data=top, x="risk_score", y="Territorio", hue="policy_cluster", dodge=False, ax=ax)
ax.set_title("Top 10 regioni per priorita di intervento, 2025")
ax.set_xlabel("Risk score")
ax.set_ylabel("")
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(FIG_DIR / "policy_priority_ranking_2025.png", dpi=150)
plt.show()"""
    ),
    md("""## 3. PIL per occupato e disoccupazione prevista"""),
    code(
        """fig, ax = plt.subplots(figsize=(11, 7))
sns.scatterplot(
    data=policy_table,
    x="gdp_per_employed_lag1",
    y="pred_unemployment_rate",
    size="pred_unemployed_thousands",
    hue="policy_cluster",
    sizes=(60, 450),
    alpha=0.82,
    legend=False,
    ax=ax,
)
for i, (_, row) in enumerate(policy_table.head(8).iterrows()):
    offset_y = 0.08 if i % 2 == 0 else -0.12
    ax.text(row["gdp_per_employed_lag1"] + 0.4, row["pred_unemployment_rate"] + offset_y, row["Territorio"], fontsize=9)
ax.axhline(policy_table["pred_unemployment_rate"].median(), color="black", linewidth=1, linestyle="--")
ax.axvline(policy_table["gdp_per_employed_lag1"].median(), color="black", linewidth=1, linestyle="--")
ax.set_title("Disoccupazione prevista vs PIL per occupato")
ax.set_xlabel("PIL per occupato 2024 (migliaia di euro)")
ax.set_ylabel("Disoccupazione prevista 2025 (%)")
ax.text(
    0.99,
    0.02,
    "Linee tratteggiate = mediane\\nPunto piu grande = piu disoccupati stimati",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC"},
)
plt.tight_layout()
plt.savefig(FIG_DIR / "policy_gdp_unemployment_quadrants.png", dpi=150)
plt.show()"""
    ),
    md("""## 4. Risposta concreta: dove agire e come"""),
    code(
        """recommendations = policy_table.head(8).copy()
recommendations["azione_consigliata"] = recommendations.apply(policy.recommend_action, axis=1)
recommendations[
    [
        "Territorio",
        "pred_unemployment_rate",
        "unemployment_rate",
        "pred_unemployed_thousands",
        "gdp_per_employed_lag1",
        "neet_rate_lag1",
        "policy_cluster",
        "azione_consigliata",
    ]
]"""
    ),
    md(
        """### Sintesi interpretativa

Le aree che richiedono piu attenzione sono quelle in cui convivono:

- disoccupazione prevista alta;
- molti disoccupati potenziali in valore assoluto;
- PIL per occupato sotto la mediana;
- NEET elevato.

Questo profilo indica che non basta una politica generica di ricerca lavoro: serve rafforzare sia la domanda di lavoro, cioe imprese e investimenti, sia l'offerta di competenze."""
    ),
    code(
        """priority_regions = recommendations["Territorio"].tolist()
print("Regioni prioritarie:", ", ".join(priority_regions[:5]))

answer = f\"\"\"
Risposta di policy:

Le prime aree su cui intervenire sono {', '.join(priority_regions[:5])}. 
Il motivo e che combinano disoccupazione prevista elevata, NEET alto e PIL per occupato relativamente basso. 
In queste regioni il problema non sembra solo congiunturale: il mercato del lavoro assorbe poco e la base produttiva e meno capace di generare occupazione stabile.

Azione consigliata:
1. usare incentivi e fondi regionali/europei per attrarre investimenti produttivi nei settori con domanda;
2. collegare formazione tecnica, ITS e apprendistato ai fabbisogni delle imprese locali;
3. rafforzare politiche attive e centri per l'impiego con target specifico sui giovani NEET;
4. monitorare annualmente se PIL per occupato e occupazione migliorano, non solo se scende il tasso di disoccupazione.
\"\"\"
print(answer)"""
    ),
    md(
        """## 5. Come raccontarlo nel progetto

Una formulazione efficace per la presentazione:

> I modelli ML non lineari prevedono meglio la disoccupazione rispetto alla regressione classica. Tuttavia, il risultato diventa utile solo se viene trasformato in priorita territoriali. Incrociando forecast della disoccupazione, PIL per occupato, NEET e numero stimato di disoccupati, emergono regioni in cui la risposta deve combinare investimenti produttivi e politiche attive del lavoro. Per gli occupati, invece, la baseline dell'anno precedente e molto forte: questo suggerisce che l'occupazione regionale e persistente e che la semplice complessita modellistica non basta a migliorare il forecast.

Questa parte risponde alla domanda: non solo "quanto prevedo", ma "dove agire e con quale leva"."""
    ),
]

nb["cells"] = cells
output = Path("notebooks/03_policy_insights.ipynb")
output.write_text(nbf.writes(nb), encoding="utf-8")
print(output)

# Big Data Project: Occupazione e Disoccupazione in Italia

Progetto di analisi su dati ISTAT regionali per studiare e poi predire:

- tasso di disoccupazione regionale;
- occupati regionali in migliaia;
- impatto del PIL regionale come fattore esterno.

## Struttura

- `data/raw/`: copie locali dei CSV originali con nomi semplificati.
- `data/processed/`: panel regione-anno generato dalle utility.
- `notebooks/01_eda_mercato_lavoro_italia.ipynb`: prima EDA eseguita.
- `notebooks/02_forecasting_models.ipynb`: confronto forecasting tra OLS classica e modelli ML.
- `notebooks/03_policy_insights.ipynb`: trasformazione dei forecast in priorita territoriali e raccomandazioni.
- `notebooks/04_validation_classification_conclusions.ipynb`: backtesting rolling, feature importance, classificazione regioni critiche e conclusione finale.
- `src/utils.py`: funzioni di caricamento, pulizia, filtro regioni e merge.
- `src/modeling.py`: funzioni per split temporale, modelli, predizioni e metriche.
- `src/policy.py`: funzioni per indice di rischio, cluster territoriali e azioni consigliate.
- `reports/figures/`: grafici esportati dal notebook.

## Ambiente

```bash
source .venv/bin/activate
jupyter notebook
```

In VS Code seleziona il kernel Python dentro `.venv`.

## Stato attuale

Sono pronte le prime due fasi:

- setup, dati copiati, utility, notebook EDA e panel `data/processed/labour_panel.csv`;
- forecasting con baseline lag-1, OLS classica, Random Forest e Gradient Boosting;
- confronto tra OLS, ML lineare e ML non lineare;
- tabella di priorita territoriale `data/processed/policy_priority_table_2025.csv`;
- backtesting rolling in `data/processed/rolling_backtest_summary.csv`;
- classificazione delle regioni critiche in `data/processed/critical_region_classification_metrics.csv`;
- output modellistici in `data/processed/forecast_metrics.csv` e `data/processed/forecast_predictions.csv`.

La prossima fase puo essere la costruzione di una presentazione o relazione finale con metodologia, risultati, limiti e raccomandazioni.

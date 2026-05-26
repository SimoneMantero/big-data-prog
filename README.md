# Big Data Project: Previsione di occupazione e disoccupazione in Italia

Progetto universitario di Big Data dedicato all'analisi e previsione del mercato del lavoro italiano su base regionale. Il lavoro integra dataset ISTAT su disoccupazione, occupazione, NEET e PIL regionale per costruire un panel regione-anno e trasformare i risultati dei modelli in indicazioni di policy territoriale.

L'obiettivo non e solo prevedere tasso di disoccupazione e numero di occupati, ma anche capire quali aree italiane risultano piu critiche e quali leve di intervento possono essere suggerite dai dati.

## Domande di ricerca

- Quanto e prevedibile la disoccupazione regionale usando dati storici, NEET e PIL?
- I modelli di Machine Learning migliorano rispetto a una regressione OLS classica?
- Per gli occupati serve davvero un modello complesso o basta la persistenza dell'anno precedente?
- Quali regioni risultano prioritarie per politiche occupazionali mirate?

## Dati utilizzati

- tasso di disoccupazione regionale;
- occupati regionali in migliaia;
- incidenza e valori NEET;
- PIL regionale lato produzione come fattore economico esterno.

## Metodologia

Il progetto segue una pipeline lineare:

1. EDA sui dataset ISTAT regionali.
2. Costruzione del panel regione-anno.
3. Feature engineering con variabili laggate.
4. Forecasting con baseline lag-1, OLS, Ridge, ElasticNet, Random Forest e Gradient Boosting.
5. Backtesting rolling per valutare la stabilita dei modelli nel tempo.
6. Classificazione delle regioni critiche.
7. Traduzione dei risultati in policy insight territoriali.

## Risultati principali

- Per la disoccupazione, il modello migliore e Random Forest, con errore medio rolling di circa 0.80 punti percentuali.
- Per gli occupati, la baseline lag-1 risulta molto competitiva: l'occupazione regionale e fortemente persistente nel tempo.
- La classificazione delle regioni critiche ottiene F1 pari a circa 0.92 con Logistic Regression.
- Le regioni prioritarie emerse dall'analisi sono Campania, Sicilia, Calabria, Puglia e Sardegna.
- Il PIL assoluto spiega bene la scala economica e occupazionale, mentre PIL per occupato e NEET aiutano a leggere la fragilita del territorio.

## Output finale

La presentazione PowerPoint finale si trova in:

- `deliverables/italian_labour_forecast_policy_deck.pptx`

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

Sono pronte tutte le fasi principali:

- setup, dati copiati, utility, notebook EDA e panel `data/processed/labour_panel.csv`;
- forecasting con baseline lag-1, OLS classica, Random Forest e Gradient Boosting;
- confronto tra OLS, ML lineare e ML non lineare;
- tabella di priorita territoriale `data/processed/policy_priority_table_2025.csv`;
- backtesting rolling in `data/processed/rolling_backtest_summary.csv`;
- classificazione delle regioni critiche in `data/processed/critical_region_classification_metrics.csv`;
- output modellistici in `data/processed/forecast_metrics.csv` e `data/processed/forecast_predictions.csv`.

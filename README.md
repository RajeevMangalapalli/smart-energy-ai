# Smart Energy AI
### AI-Driven Energy Forecasting and Explanation System

A production-structured machine learning project that forecasts German electricity demand using time series and ML models, and explains those forecasts using a Retrieval-Augmented Generation (RAG) system powered by a local LLM.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [Quickstart](#quickstart)
- [Pipeline Modules](#pipeline-modules)
- [Model Results](#model-results)
- [RAG System](#rag-system)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Running Tests](#running-tests)
- [Wiki](#wiki)
- [Dependencies](#dependencies)

---

## Overview

This project combines two AI systems into one integrated pipeline:

1. **Forecasting Engine** — A SARIMA baseline and an XGBoost ML model trained on 5 years of German daily electricity load data. The XGBoost model achieves a **4.6% RMSE** relative to mean load, compared to **13% for SARIMA**.

2. **RAG Explanation System** — A retrieval-augmented generation pipeline that retrieves relevant knowledge base chunks and uses a local Mistral LLM (via Ollama) to generate natural language explanations of the forecast, with source citations.

---

## Project Structure

```
smart-energy-ai/
├── data/
│   ├── external/
│   ├── processed/
│   │   ├── DE_load_daily.csv
│   │   ├── DE_load_daily_features_time_features.csv
│   │   ├── sarima_predictions.csv
│   │   ├── sarima_metrics.json
│   │   ├── xgb_predictions.csv
│   │   ├── xgb_metrics.json
│   │   ├── forecast_comparison.png
│   │   ├── chunk_mapping.json
│   │   └── vectorstore/
│   └── raw/
├── notebooks/
│   └── 01_eda.ipynb
├── rag/
│   ├── documents/
│   │   ├── dataset_description.txt
│   │   ├── energy_forecasting_concepts.txt
│   │   └── model_assumptions_and_features.txt
│   ├── ingest.py
│   ├── retrieve.py
│   └── pipeline.py
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   └── features.py
│   └── models/
│       ├── baseline.py
│       ├── ml_model.py
│       └── evaluate.py
└── README.md
```

---

## System Architecture

```
Raw CSV Data
     │
     ▼
ingestion.py ──► features.py ──► DE_load_daily_features_time_features.csv
                                          │
                      ┌───────────────────┴──────────────────┐
                      ▼                                       ▼
               baseline.py                             ml_model.py
             SARIMA(0,0,3)(0,0,2,7)                  XGBoost Regressor
               RMSE: 13%                               RMSE: 4.6%
                      │                                       │
                      └───────────────┬───────────────────────┘
                                      ▼
                                evaluate.py
                            (comparison plots)
                                      │
                                      ▼
                              pipeline.py
                         (forecast next 7 days)
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │         RAG SYSTEM               │
                    │                                  │
                    │  ingest.py → ChromaDB            │
                    │  retrieve.py → Mistral (Ollama)  │
                    │  Forecast + Explanation + Citations│
                    └─────────────────────────────────┘
```

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/smart-energy-ai.git
cd smart-energy-ai
pip install -r requirements.txt
```

### 2. Install and start Ollama

Download from https://ollama.com, then:

```bash
ollama pull mistral
ollama serve          # run this in a separate terminal
```

### 3. Run data pipeline

```bash
python src/data/ingestion.py
python src/data/features.py
```

### 4. Build the knowledge base

```bash
python rag/ingest.py
```

### 5. Run the full pipeline

```bash
python rag/pipeline.py
```

---

## Pipeline Modules

### `src/data/ingestion.py`
Loads the raw CSV, handles missing values, aligns timestamps, and outputs a clean daily time series.

### `src/data/features.py`
Engineers lag features (`lag_1`, `lag_7`) and rolling means (`rolling_mean_7`, `rolling_mean_30`) from the cleaned data. These features are the primary inputs to the XGBoost model.

### `src/models/baseline.py`
Fits a SARIMA model with order `(0,0,3)(0,0,2,7)` on the training split. The seasonal component `s=7` captures the weekly cycle visible in the ACF/PACF plots. Saves predictions and metrics to `data/processed/`.

### `src/models/ml_model.py`
Trains an XGBoost regressor on lag features, rolling means, and calendar features (day of week, month, is_weekend). Uses a time-based train/test split to prevent data leakage. Saves predictions and metrics to `data/processed/`.

### `src/models/evaluate.py`
Loads both sets of predictions and produces a comparison plot (actuals vs. SARIMA vs. XGBoost) with a residuals panel, and a metrics summary table.

### `rag/ingest.py`
Loads the three knowledge base documents, chunks them into ~300 character pieces with 50 character overlap, embeds each chunk using `sentence-transformers/all-MiniLM-L6-v2`, and stores the result in a persistent ChromaDB vector store.

### `rag/retrieve.py`
Provides the full RAG pipeline: embeds a user query, retrieves the top-3 most similar chunks from ChromaDB by cosine similarity, checks relevance against a distance threshold, and calls Mistral via Ollama to generate a cited answer.

### `rag/pipeline.py`
Integrates forecasting and RAG. Retrains XGBoost on the full dataset, forecasts the next 7 days iteratively, formats the forecast as a natural language summary, and passes the user's question (with forecast context) to the RAG system to generate an explanation.

---

## Model Results

| Model | RMSE (MW) | RMSE % of Mean |
|---|---|---|
| SARIMA(0,0,3)(0,0,2,7) | ~7,249 | 13% |
| XGBoost | ~2,500 | 4.6% |

The XGBoost model outperforms SARIMA by nearly 3x because it directly uses engineered lag and calendar features that SARIMA must approximate through seasonal terms. The weekly seasonality (visible in ACF/PACF at lag 7, 14, 21...) is captured by `lag_7` and `is_weekend` features in XGBoost, versus explicit seasonal order parameters in SARIMA.

---

## RAG System

The knowledge base contains three documents:

| Document | Contents |
|---|---|
| `dataset_description.txt` | Source, time range, column definitions for the German load dataset |
| `energy_forecasting_concepts.txt` | Types of forecasting, demand drivers, accuracy importance |
| `model_assumptions_and_features.txt` | Engineered features, model assumptions, feature utility |

Chunking produces 18 chunks across the three documents. Each chunk is embedded with `all-MiniLM-L6-v2` (384-dimensional vectors) and stored in ChromaDB with cosine similarity indexing.

The retrieval threshold is set at a cosine distance of `0.7`. Queries with no chunk below this threshold receive a refusal response rather than a hallucinated answer.

---

## Running the Full Pipeline

```bash
# Make sure ollama is serving in a separate terminal
ollama serve

# Run the integrated forecast + explanation pipeline
python rag/pipeline.py
```

Example output:

```
═══════════════════════════════════════════════════════
 SMART ENERGY AI — FORECAST + EXPLANATION PIPELINE
═══════════════════════════════════════════════════════
Model trained on 2070 days of data

Forecast for the next 7 days (German electricity demand):
  - Thursday 2020-10-01: 56,575 MW
  - Friday 2020-10-02: 54,965 MW
  - Saturday 2020-10-03: 46,334 MW
  - Sunday 2020-10-04: 43,241 MW
  - Monday 2020-10-05: 53,624 MW
  - Tuesday 2020-10-06: 55,216 MW
  - Wednesday 2020-10-07: 56,555 MW

Weekly average : 52,359 MW
Peak day       : Thursday at 56,575 MW
Lowest day     : Sunday at 43,241 MW

EXPLANATION FROM KNOWLEDGE BASE
──────────────────────────────────────────────────
The model uses lag features (lag_1, lag_7) and rolling means to capture
autocorrelation and weekly cycles in electricity demand...

Sources: model_assumptions_and_features.txt, energy_forecasting_concepts.txt
```

---

## Wiki

Detailed documentation is available in the project wiki:

- [Wiki 1 — Data Pipeline and Feature Engineering](https://github.com/RajeevMangalapalli/smart-energy-ai/wiki)
- [Wiki 2 — Time Series Analysis and Model Selection](https://github.com/RajeevMangalapalli/smart-energy-ai/wiki/Time-Series-Analysis-and-Model-Selection)
- [Wiki 3 — RAG System Architecture](https://github.com/RajeevMangalapalli/smart-energy-ai/wiki/RAG-SYSTEM)
- [Wiki 4 — Running and Extending the Project](https://github.com/RajeevMangalapalli/smart-energy-ai/wiki/Extending-the-project)

---

## Dependencies

```
pandas
numpy
statsmodels
pmdarima
xgboost
scikit-learn
matplotlib
langchain-text-splitters
sentence-transformers
chromadb
ollama
pytest
```

Install all with:

```bash
pip install -r requirements.txt
```

Ollama must be installed separately from https://ollama.com. Pull the Mistral model with `ollama pull mistral`.

## AI DRIVEN ENERGY FORECASTING AND EXPLANATION SYSTEM (RAG+TIME SERIES)

---

## Description

This project attempts to capture the following features/ascpects:

---
1) DATA INGESTION AND PREPROCESSING

- Load a real-world energy time-series dataset
- Handle : missing values, time alignment, train/validation split
- Feature Engineering : lag features, rolling means
- Output : A dataset pipeline that is reproducible + Saved preprocessed dataset

---
2) TIME SERIES FORECASTING ENGINE

- Baseline model : ARIMA which is used as a reference point
- ML Model : XGBoost, trained on the same data
- Evaluation : Metrics (MAE, RMSE); Compare the baseline vs ML and store predictions in a structured format
- Output : Forecast values for the next N days

---
3) KNOWLEDGE BASE FOR EXPLANATIONS (RAG SYSTEM)

- Create a document corpus containing : dataset description, forecasting assumptions, short energy domain articles (PDF/text)
- Chunk documents consistently
- Generate embeddings
- Store in a vector database
- Output : Vector index persisted locally; Mapping from chunks -> source documents

---
4) RAG SYSTEM

- Retrieve top-k relevant chunks based on user query
- Inject retrieved context into LLM prompt
- Enforce : retrieval only answers; refusal of no relevant context is found
- Include citations in the final answer
- Output : Natural Language explanation; list of cited sources

---
5) FORECAST + EXPLANATION INTEGRATION

- Connection of forecasting output to RAG system
---


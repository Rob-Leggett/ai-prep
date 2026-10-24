# 🧠 ai-prep

A **local AI playground** combining:

- **Classic ML (CatBoost + MLflow)**
- **Retrieval-Augmented Generation (RAG)**
- **Local LLM orchestration (Ollama + LangChain + LangGraph)**
- **FastAPI serving**
- **Lightweight “lakehouse” data handling with DuckDB**

---

## 🚀 Project Overview

This project demonstrates how to integrate structured ML models, document retrieval, and local LLM reasoning into a single unified workflow — completely offline on your Mac.

**Structure**

```
├─ app/                      # FastAPI serving layer
│  ├─ main.py                # /predict (CatBoost), /ask (RAG), /agent (LangGraph)
├─ ml/
│  ├─ train_catboost.py      # trains & logs CatBoost model via MLflow
│  └─ register_model.py      # (optional) local registry pattern
├─ rag/
│  ├─ build_index.py         # chunk + embed PDFs into the local vector store
│  ├─ build_index_from_csv.py# embed structured CSV into the local vector store
│  ├─ store.py               # DuckDB-backed vector store (cosine similarity)
│  └─ query.py               # test queries with citations
├─ mcp_server.py             # MCP server (mcp 2.x MCPServer) exposing the API as tools
├─ tests/                    # pytest suite
├─ lakehouse/
│  └─ legal/...              # parquet/csv “domains” for DuckDB
├─ data/                     # sample CSVs and PDFs
└─ README.md
```

---

## 🧩 Environment Setup

Requires **Python 3.13+** and [uv](https://docs.astral.sh/uv/) (`brew install uv`).

### Install dependencies
```bash
uv sync                      # creates .venv from uv.lock (app + dev tools)
source .venv/bin/activate    # or prefix commands with `uv run`
```

### Lint, test and audit
```bash
make test                    # ruff check . && pytest && pip-audit
```

### Upgrade dependencies
```bash
uv lock --upgrade && uv sync
```

### Makefile

| **Command**          | **Description**                                                                                       |
|----------------------|-------------------------------------------------------------------------------------------------------|
| `make install`       | Create the virtual environment and install dependencies (`uv sync`).                                  |
| `make test`          | Run ruff, pytest and pip-audit.                                                                       |
| `make serve`         | Start the FastAPI app (background, with reload) at http://127.0.0.1:8000.                             |
| `make mlflow`        | Launch the MLflow tracking UI (background) at http://127.0.0.1:5000.                                  |
| `make ollama-model`  | Ensure a tools-capable Ollama model (`llama3.1`) is downloaded.                                       |
| `make ollama`        | Start the Ollama runtime (background) at http://127.0.0.1:11434.                                      |
| `make index-csv`     | Build the CSV-based RAG index using `rag/build_index_from_csv.py` (currently waits on app port).      |
| `make mcp-all`       | Start FastAPI, then the MCP server (`mcp_server.py`) for IDE/editor tool integration.                 |
| `make run-all`       | Pull model, start Ollama, FastAPI, and MLflow, then build the CSV index.                              |
| `make stop`          | Stop background services (FastAPI, MLflow, Ollama) and clean up PIDs.                                 |
| `make clean-pids`    | Remove PID files from `.pids/`.                                                                       |
| `make train`         | Train CatBoost, log to MLflow, and pin latest model URI to `.env`.                                    |
| `make pin-latest`    | Re-pin `.env` to the most recent MLflow run without retraining.                                       |
| `make model-uri`     | Show the currently pinned MLflow model URI from `.env`.                                               |

---

## 🧪 Train Your CatBoost Model

```bash
python ml/train_catboost.py
```

Expected output:
```
Logged model. MAE=0.0410 R2=0.8194 run_id=ba91400a5aaf4a4ba5528efd7128f569
```

This logs your model to the local MLflow store (`mlflow.db` for run metadata, `mlruns/` for artefacts).

---

## 📊 Run the MLflow UI

```bash
mlflow ui
```

Access it at:  
➡️ [http://127.0.0.1:5000](http://127.0.0.1:5000)

## ⚙️ Run MCP Server

```bash
make mcp-all
```

### Connect a Client (required)

```bash
brew install uv

mcp dev mcp_server.py # install what you are asked to install
```

🚀 MCP Inspector is up and running at:  http://localhost:6274/?MCP_PROXY_AUTH_TOKEN=<AUTOMATED_GENERATED_TOKEN>

- Select `Connect`
- Select `Tools`
- Select `List Tools`
- Select `Health`
- Select `Run Tool`

Your tool is now running via MCP

---

## 🦙 Install and Start Ollama (Required for `/ask` and `/agent`)

```bash
brew install ollama
ollama serve
# or run it in the background at login:
brew services start ollama

# Pull a model
ollama pull llama3.1

# Verify
ollama list
curl http://localhost:11434/api/tags
```

If Ollama isn’t running, `/ask` will fail with “Connection refused”.

---

## 🔎 Build Your RAG Index

### From PDFs
```bash
python rag/build_index.py
```

### From CSV (sample structured data)
```bash
python rag/build_index_from_csv.py
```

Expected output:
```
✅ Indexed 100 rows from sample.csv into 'legal_docs'.
```

---

## ⚙️ Run the FastAPI App

```bash
uvicorn app.main:app --reload
```

Available endpoints:

| **Endpoint**   | **Description**                       | **Example**                                                                                                                                                      |
|----------------|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `/`            | Redirects to interactive docs         | [http://127.0.0.1:8000/](http://127.0.0.1:8000/)                                                                                                                 |
| `/health`      | Health check                          | [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)                                                                                                     |
| `/predict`     | Calls CatBoost model                  | [http://127.0.0.1:8000/predict?age=42&income=90000](http://127.0.0.1:8000/predict?age=42&income=90000)                                                           |
| `/ask`         | Queries your RAG index via Llama 3.1  | [http://127.0.0.1:8000/ask?question=Which%20age%20has%20the%20highest%20income?](http://127.0.0.1:8000/ask?question=Which%20age%20has%20the%20highest%20income?) |
| `/agent`       | Orchestrates ML + RAG via LangGraph   | [http://127.0.0.1:8000/agent?query=Which%20tool%20should%20I%20use?](http://127.0.0.1:8000/agent?query=Which%20tool%20should%20I%20use?)                         |

Interactive Swagger UI:  
➡️ [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧮 Pipeline Summary

| **Step**           | **Command**                                                           | **Result**                         |
|--------------------|-----------------------------------------------------------------------|------------------------------------|
| 1. Train CatBoost  | `python ml/train_catboost.py`                                         | Logs model in `mlruns/`            |
| 2. Build RAG index | `python rag/build_index.py` or `python rag/build_index_from_csv.py`   | Indexes PDFs / CSV                 |
| 3. Start API       | `uvicorn app.main:app --reload`                                       | `/predict`, `/ask`, `/agent` live  |
| 4. Try query       | Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)        | Interactive Swagger UI             |

---

## 🧰 Tech Stack Overview

| **Category**        | **Key Libraries**                                | **Purpose**                                    |
|---------------------|--------------------------------------------------|------------------------------------------------|
| Classic ML          | `catboost`, `shap`                               | Model training + interpretability              |
| Serving             | `fastapi`, `uvicorn`                             | API endpoints for `/predict`, `/ask`, `/agent` |
| RAG Stack           | `duckdb`, `sentence-transformers`, `faiss-cpu`   | Embed and query structured/unstructured data   |
| Agents              | `langchain`, `langgraph`, `langchain-ollama`     | Build reasoning chains using multiple tools    |
| LLM Runtime         | `ollama` *(external binary)*                     | Run Llama 3.1 locally                          |
| Data                | `duckdb`, `pyarrow`                              | Mini-lakehouse for structured datasets         |
| Experiment Tracking | `mlflow`                                         | Track metrics, artefacts, and models           |
| Visual / Notebook   | `jupyterlab`, `matplotlib`, `seaborn`            | Exploration & plotting                         |

---

## 💾 Sample DuckDB Query

```python
import duckdb
df = duckdb.query("SELECT * FROM 'lakehouse/legal/cases.parquet' LIMIT 10").to_df()
print(df.head())
```

---

## 🧹 .gitignore Recommendations

```gitignore
# Python / venv
.venv/
__pycache__/
*.pyc

# MLflow / CatBoost
mlruns/
mlartifacts/
catboost_info/

# Vector DB / RAG
rag/vector_store.duckdb*
mlflow.db

# Data
data/*.csv
data/*.parquet

# Jupyter
.ipynb_checkpoints/
```

---

## ✅ Quick Recap

| **Feature**         | **Runs on**                     | **Notes**                                      |
|---------------------|---------------------------------|------------------------------------------------|
| CatBoost training   | `ml/train_catboost.py`          | Logs to MLflow                                 |
| MLflow tracking UI  | `mlflow ui`                     | [http://127.0.0.1:5000](http://127.0.0.1:5000) |
| Local LLM (Llama 3.1) | `ollama serve`                  | No cloud dependency                            |
| RAG index           | `rag/build_index*.py`           | Embeds CSV & PDFs                              |
| API layer           | `uvicorn app.main:app --reload` | `/predict`, `/ask`, `/agent`                   |

---

### 🏁 Example Workflow

```bash
# Setup
uv sync
source .venv/bin/activate

# Train + track
python ml/train_catboost.py
mlflow ui &

# Build RAG
python rag/build_index_from_csv.py

# Run Ollama
ollama serve &

# Launch API
uvicorn app.main:app --reload
```

Then open your browser to:  
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### ✨ Author
**Robert Leggett**  
<contact@robertleggett.com.au>  
Built for local AI experimentation and preparation for production-grade hybrid ML + LLM pipelines.

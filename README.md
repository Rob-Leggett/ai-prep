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
│  ├─ build_index.py         # chunk + embed PDFs into Chroma
│  ├─ build_index_from_csv.py# embed structured CSV into Chroma
│  └─ query.py               # test queries with citations
├─ lakehouse/
│  └─ legal/...              # parquet/csv “domains” for DuckDB
├─ data/                     # sample CSVs and PDFs
└─ README.md
```

---

## 🧩 Environment Setup

### 1️⃣ Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2️⃣ Install dependencies
```bash
pip install -e ".[dev]"
```

If you see build errors for `chromadb` on macOS, install Rust first:
```bash
brew install rust
```

---

## 🧪 Train Your CatBoost Model

```bash
python ml/train_catboost.py
```

Expected output:
```
Logged model. MAE=0.0410 R2=0.8194 run_id=ba91400a5aaf4a4ba5528efd7128f569
```

This logs your model to the local MLflow tracking directory (`mlruns/`).

---

## 📊 Run the MLflow UI

```bash
mlflow ui
```

Access it at:  
➡️ [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🦙 Install and Start Ollama (Required for `/ask` and `/agent`)

```bash
brew install ollama
ollama serve
# or run it in the background at login:
brew services start ollama

# Pull a model
ollama pull llama3

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
| `/ask`         | Queries your Chroma index via Llama 3 | [http://127.0.0.1:8000/ask?question=Which%20age%20has%20the%20highest%20income?](http://127.0.0.1:8000/ask?question=Which%20age%20has%20the%20highest%20income?) |
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
| RAG Stack           | `chromadb`, `sentence-transformers`, `faiss-cpu` | Embed and query structured/unstructured data   |
| Agents              | `langchain`, `langgraph`, `langchain-ollama`     | Build reasoning chains using multiple tools    |
| LLM Runtime         | `ollama` *(external binary)*                     | Run Llama 3 locally                            |
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
rag/chroma_store/

# Data
data/*.csv
data/*.parquet

# Jupyter
.ipynb_checkpoints/
```

---

## ✅ Quick Recap

| **Feature** | **Runs on** | **Notes** |
|--------------|-------------|------------|
| CatBoost training | `ml/train_catboost.py` | Logs to MLflow |
| MLflow tracking UI | `mlflow ui` | [http://127.0.0.1:5000](http://127.0.0.1:5000) |
| Local LLM (Llama 3) | `ollama serve` | No cloud dependency |
| RAG index | `rag/build_index*.py` | Embeds CSV & PDFs |
| API layer | `uvicorn app.main:app --reload` | `/predict`, `/ask`, `/agent` |

---

### 🏁 Example Workflow

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

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

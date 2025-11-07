import os
import mlflow
from dotenv import load_dotenv
import pandas as pd
from rag.query import query_rag

# Load once at import (reuse across calls)
load_dotenv()  # reads .env

model_uri = os.getenv("MLFLOW_MODEL_URI")

if not model_uri:
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=["0"], order_by=["attribute.start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError("No MLflow runs found — train a model first.")
    model_uri = f"runs:/{runs[0].info.run_id}/model"

print(f"Loading model from {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

def predict_income(age: float, income: float) -> float:
    df = pd.DataFrame([{"age": float(age), "income": float(income)}], dtype="float64")
    return float(model.predict(df)[0])

def ask_docs(question: str) -> tuple[str, list[dict]]:
    answer, sources = query_rag(question)
    return answer, sources
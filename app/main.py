from fastapi import FastAPI, Query
import mlflow
import pandas as pd
from starlette.responses import JSONResponse, RedirectResponse

from rag.query import query_rag
from langchain_ollama import OllamaLLM

app = FastAPI(title="Local AI Platform")

# Load CatBoost model logged in MLflow
model = mlflow.pyfunc.load_model("runs:/a1b7209b0c6e487fb2fd84e5a45795de/model")

@app.get("/", include_in_schema=False)
def home():
    # send people to the Swagger UI
    return RedirectResponse(url="/docs")

@app.get("/health", include_in_schema=False)
def health():
    return JSONResponse({"status": "ok"})

@app.get("/predict")
def predict(age: int, income: float):
    df = pd.DataFrame([{"age": age, "income": income}], dtype="float64")
    pred = model.predict(df)[0]
    return JSONResponse({"prediction": float(pred)})

@app.get("/ask")
def ask(question: str = Query(..., description="Ask about your documents")):
    answer, sources = query_rag(question)
    return JSONResponse({"answer": answer, "sources": sources})

@app.get("/agent")
def agent_flow(query: str):
    llm = OllamaLLM(model="llama3")
    prompt = f"Given user query '{query}', decide if you should predict (numerical) or retrieve (textual)."
    decision = llm.invoke(prompt)
    return JSONResponse({"decision": decision})
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse, JSONResponse
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from app.core import predict_income, ask_docs
from app.tools import predict_income_tool, ask_docs_tool

app = FastAPI(title="Local AI Platform")

@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")

@app.get("/health", include_in_schema=False)
def health():
    return JSONResponse({"status": "ok"})

@app.get("/predict")
def predict(age: float, income: float):
    # age is float to match model signature; we cast inside core anyway
    pred = predict_income(age, income)
    return {"prediction": pred}

@app.get("/ask")
def ask(question: str = Query(..., description="Ask about your documents")):
    answer, sources = ask_docs(question)
    return {"answer": answer, "sources": sources}

# Agent with the same business logic as tools
LLM = ChatOllama(model="llama3.1", temperature=0)
TOOLS = [predict_income_tool, ask_docs_tool]
AGENT = create_agent(LLM, TOOLS)

@app.get("/agent")
def agent(query: str = Query(..., description="User query")):
    result = AGENT.invoke({"messages": [HumanMessage(content=query)]})
    final = result["messages"][-1].content
    return {"result": final}
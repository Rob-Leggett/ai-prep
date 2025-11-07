from pydantic import BaseModel, Field
from langchain.tools import tool
from app.core import predict_income, ask_docs

class PredictArgs(BaseModel):
    age: float = Field(..., description="Age in years")
    income: float = Field(..., description="Annual income")

class PredictResult(BaseModel):
    prediction: float

@tool("predict_income", args_schema=PredictArgs, return_direct=False)
def predict_income_tool(age: float, income: float) -> dict:
    """Run CatBoost prediction for a person with age and income."""
    pred = predict_income(age, income)
    return PredictResult(prediction=pred).model_dump()

class AskArgs(BaseModel):
    question: str = Field(..., description="Natural language question for the document index")

class AskResult(BaseModel):
    answer: str
    sources: list[dict]

@tool("ask_docs", args_schema=AskArgs, return_direct=False)
def ask_docs_tool(question: str) -> dict:
    """Query the RAG index and return an answer with sources."""
    answer, sources = ask_docs(question)
    return AskResult(answer=answer, sources=sources).model_dump()
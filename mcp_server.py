import asyncio
import os
import sys

import requests
from mcp.server.mcpserver import MCPServer

APP_NAME = "ai-prep"
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

mcp = MCPServer(APP_NAME)

def _log(msg: str):
    print(f"[MCP:{APP_NAME}] {msg}", file=sys.stderr, flush=True)

@mcp.tool()
def predict(age: float, income: float) -> dict:
    """Run CatBoost prediction via FastAPI /predict."""
    r = requests.get(f"{API_BASE}/predict", params={"age": age, "income": income}, timeout=10)
    r.raise_for_status()
    return r.json()

@mcp.tool()
def ask(question: str) -> dict:
    """Query the local RAG index via FastAPI /ask."""
    r = requests.get(f"{API_BASE}/ask", params={"question": question}, timeout=15)
    r.raise_for_status()
    return r.json()

@mcp.tool()
def health() -> dict:
    """Check FastAPI health (no arguments)."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        ok = (r.status_code == 200)
        return {"ok": ok, "body": r.json() if ok else r.text}
    except (requests.RequestException, ValueError) as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    _log(f"starting (API_BASE={API_BASE})")
    tool_names = sorted(t.name for t in asyncio.run(mcp.list_tools()))
    _log(f"tools registered: {', '.join(tool_names)}")
    mcp.run()  # stdio server; stays idle until a client connects
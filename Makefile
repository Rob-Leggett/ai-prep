SHELL := /bin/bash
.ONESHELL:
PY := .venv/bin/python
PIP := .venv/bin/pip
UVICORN := .venv/bin/uvicorn

APP_HOST ?= 127.0.0.1
APP_PORT ?= 8000
MLFLOW_PORT ?= 5000
OLLAMA_HOST ?= 127.0.0.1
OLLAMA_PORT ?= 11434

PIDDIR := .pids
APP_PID := $(PIDDIR)/app.pid
ML_PID  := $(PIDDIR)/mlflow.pid
OL_PID  := $(PIDDIR)/ollama.pid
MCP_PID  := $(PIDDIR)/mcp.pid

.PHONY: install serve mlflow ollama index-csv run-all mcp stop clean-pids

# --- Install / Setup ---
install:
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -U pip setuptools wheel
	pip install -e ".[dev]"

# --- Servers ---

serve:
	mkdir -p $(PIDDIR)
	($(UVICORN) app.main:app --host $(APP_HOST) --port $(APP_PORT) --reload > .app.log 2>&1 & echo $$! > $(APP_PID))
	@echo "FastAPI → http://$(APP_HOST):$(APP_PORT)"

mlflow:
	mkdir -p $(PIDDIR)
	($(PY) -m mlflow ui --host 127.0.0.1 --port $(MLFLOW_PORT) > .mlflow.log 2>&1 & echo $$! > $(ML_PID))
	@echo "MLflow → http://127.0.0.1:$(MLFLOW_PORT)"

ollama-model:
	@echo "📦 Ensuring tools-capable LLM (llama3.1) is available..."
	ollama pull llama3.1
	@echo "✅ llama3.1 ready for LangGraph agent"

ollama:
	mkdir -p $(PIDDIR)
	(OLLAMA_HOST=http://$(OLLAMA_HOST):$(OLLAMA_PORT) ollama serve > .ollama.log 2>&1 & echo $$! > $(OL_PID))
	@echo "Ollama → http://$(OLLAMA_HOST):$(OLLAMA_PORT)"

wait-port = bash -lc 'for i in {1..60}; do nc -z $(1) $(2) && exit 0; sleep 1; done; echo "Timeout waiting for $(1):$(2)"; exit 1'

index-csv:
	@$(call wait-port,$(APP_HOST),$(APP_PORT)); \
	$(PY) rag/build_index_from_csv.py

mcp-all:
	mkdir -p $(PIDDIR)
	$(MAKE) serve &
	@$(call wait-port,$(APP_HOST),$(APP_PORT))
	@echo "🚀 Starting MCP server..."
	@(PYTHONUNBUFFERED=1 $(PY) -u mcp_server.py > .mcp.log 2>&1 & echo $$! > $(PIDDIR)/mcp.pid)
	@echo "✅ API + MCP up.  Docs: http://$(APP_HOST):$(APP_PORT)/docs"

run-all:
	$(MAKE) ollama-model
	$(MAKE) -j3 ollama serve mlflow
	@$(call wait-port,$(OLLAMA_HOST),$(OLLAMA_PORT))
	@$(call wait-port,127.0.0.1,$(MLFLOW_PORT))
	@$(call wait-port,$(APP_HOST),$(APP_PORT))
	$(MAKE) index-csv
	@echo "✅ All services running:"
	@echo "   • FastAPI  → http://$(APP_HOST):$(APP_PORT)"
	@echo "   • MLflow   → http://127.0.0.1:$(MLFLOW_PORT)"
	@echo "   • Ollama   → http://$(OLLAMA_HOST):$(OLLAMA_PORT)"

stop:
	@echo "🛑 Stopping all services..."
	-@[ -f $(APP_PID) ] && kill $$(cat $(APP_PID)) && rm -f $(APP_PID) || pkill -f "uvicorn" || true
	-@[ -f $(ML_PID) ]  && kill $$(cat $(ML_PID))  && rm -f $(ML_PID)  || pkill -f "mlflow"  || true
	-@[ -f $(OL_PID) ]  && kill $$(cat $(OL_PID))  && rm -f $(OL_PID)  || pkill -f "ollama"  || true
	-@[ -f $(MCP_PID) ]  && kill $$(cat $(MCP_PID))  && rm -f $(MCP_PID)  || pkill -f "mcp_server"  || true
	@echo "✅ All processes stopped (or none were running)."

clean-pids:
	rm -f $(PIDDIR)/*.pid

# --- Train & pin latest model to .env ---
train:
	@echo "🏋️  Training CatBoost and logging to MLflow…"
	@$(PY) ml/train_catboost.py | tee .train.log
	@run_id=$$(grep -Eo 'run_id=[0-9a-f-]+' .train.log | tail -1 | cut -d= -f2); \
	if [ -n "$$run_id" ]; then \
	  echo "MLFLOW_MODEL_URI=runs:/$$run_id/model" > .env; \
	  echo "📌 Pinned latest model to .env → runs:/$$run_id/model"; \
	else \
	  echo "⚠️ Couldn’t detect run_id from training output. Leaving .env unchanged."; \
	fi

# (Optional) Re-pin to the most recent run in the default experiment (even if you didn’t just train)
pin-latest:
	@$(PY) - <<'PY'
	import mlflow, sys, os
	c = mlflow.tracking.MlflowClient()
	runs = c.search_runs(experiment_ids=["0"], order_by=["attribute.start_time DESC"], max_results=1)
	if not runs:
		print("No runs found – train first."); sys.exit(1)
	rid = runs[0].info.run_id
	open(".env","w").write(f"MLFLOW_MODEL_URI=runs:/{rid}/model\n")
	print(f"Pinned .env → runs:/{rid}/model")
	PY

# (Optional) Show what you’re pinned to
model-uri:
	@[ -f .env ] && grep '^MLFLOW_MODEL_URI=' .env || echo "No .env or MLFLOW_MODEL_URI set."
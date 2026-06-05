# Gemma Chatbot Backend

Backend-only FastAPI chatbot for Gemma, with a hybrid router:

- Simple fast path for normal chat
- LangGraph path only for complex cases
- RAG path only for explicit file/document/reference requests

Production Python code does not contain fixed chatbot replies. Chatbot behavior lives in prompt files under `app/prompts/v1`.

## Default Local Run

Docker is not required for normal development.

```bash
conda activate gemma-chatbot
pip install -r requirements.txt
cp .env.example .env
nano .env
python scripts/validate_env.py
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Set `GITHUB_TOKEN` in `.env` when using the default GitHub Models provider.

## Default Environment

```dotenv
SIMPLE_CHAT_MODE=true
ENABLE_LANGGRAPH=true
ENABLE_RESOURCE_RAG=true
ENABLE_RAG=false
ENABLE_MEMORY=false
ENABLE_DB=false
ENABLE_REDIS=false

LLM_PROVIDER=github_models
GITHUB_TOKEN=
GITHUB_MODELS_BASE_URL=https://models.github.ai/inference
LLM_MODEL=openai/gpt-4.1-mini
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=0
LLM_MAX_TOKENS=180
LLM_TEMPERATURE=0.4

HYBRID_ROUTER_MODE=deterministic
COMPLEXITY_THRESHOLD=0.65

RESOURCE_RAG_MAX_DEFAULT=1
RESOURCE_RAG_MAX_EXPLICIT=3
RESOURCE_RAG_STRONG_MATCH_THRESHOLD=0.78
RESOURCE_RAG_PERMISSION_THRESHOLD=0.55
RESOURCE_RAG_MIN_TURNS_BEFORE_SUGGEST=3
RESOURCE_RAG_MIN_DEPTH_BEFORE_RECOMMEND=0.75
RESOURCE_RAG_COOLDOWN_TURNS=5
RESOURCE_INTENT_CLASSIFIER=semantic
RESOURCE_INTENT_CLASSIFIER_MODEL=use_default_llm
RESOURCE_INTENT_CONFIDENCE_THRESHOLD=0.65
```

Resource and conversation policy live in:

- `app/config/resource_rag_policy.yml`
- `app/config/conversation_policy.yml`

## Routing

`POST /api/v1/chat` returns `path_used` and `route_reason`.

Simple path:
- Casual chat
- Short advice
- Unclear short messages
- Simple emotional support
- Simple sleep/advice messages
- Normal knowledge questions
- Normal back-and-forth

LangGraph path:
- Explicit analysis requests
- Plan or step-by-step requests
- Strategy requests
- Decision problems with multiple constraints
- Long repeated emotional/sleep problems
- Complex coaching
- Long messages or competing goals

RAG path:
- Only explicit file/document/reference wording
- Requires `ENABLE_RAG=true`
- Does not run for normal chat

If `ENABLE_RAG=false`, explicit document requests stay on the simple LLM path and the model is instructed not to invent file context.

## Optional Docker

```bash
docker compose --env-file .env up --build
```

Docker includes `host.docker.internal:host-gateway` for local OpenAI-compatible providers such as Ollama. With GitHub Models, Ollama is not required.

## Validation

```bash
pytest -q
python -m ruff check .
python -m compileall app tests alembic scripts
```

Run a live smoke test:

```bash
BASE_URL=http://127.0.0.1:8000 python scripts/smoke_test.py
```

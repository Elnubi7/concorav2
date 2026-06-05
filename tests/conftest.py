import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("APP_NAME", "gemma-test")
os.environ.setdefault("APP_VERSION", "0.1.0")
os.environ.setdefault("API_HOST", "127.0.0.1")
os.environ.setdefault("API_PORT", "8000")
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("GITHUB_TOKEN", "")
os.environ.setdefault("GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference")
os.environ.setdefault("LLM_API_KEY", "")
os.environ.setdefault("LLM_PROVIDER", "test")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("LLM_BASE_URL", "")
os.environ.setdefault("LLM_TIMEOUT_SECONDS", "5")
os.environ.setdefault("LLM_MAX_RETRIES", "0")
os.environ.setdefault("LLM_MAX_TOKENS", "120")
os.environ.setdefault("LLM_TEMPERATURE", "0.4")
os.environ["SIMPLE_CHAT_MODE"] = "true"
os.environ["HYBRID_ROUTER_MODE"] = "deterministic"
os.environ["COMPLEXITY_THRESHOLD"] = "0.65"
os.environ["ENABLE_RAG"] = "false"
os.environ["ENABLE_RESOURCE_RAG"] = "true"
os.environ["ENABLE_MEMORY"] = "false"
os.environ["ENABLE_LANGGRAPH"] = "true"
os.environ["ENABLE_DB"] = "true"
os.environ["ENABLE_REDIS"] = "false"
os.environ.setdefault("ALWAYS_FAST_CHAT", "true")
os.environ.setdefault("LLM_FAST_MAX_TOKENS", "120")
os.environ.setdefault("LLM_FAST_TEMPERATURE", "0.4")
os.environ.setdefault("LLM_FAST_TIMEOUT_SECONDS", "30")
os.environ.setdefault("EMBEDDING_PROVIDER", "test")
os.environ.setdefault("EMBEDDING_MODEL", "test-embedding")
os.environ.setdefault("EMBEDDING_API_KEY", "")
os.environ.setdefault("EMBEDDING_BASE_URL", "")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "16")
os.environ.setdefault("VECTOR_BACKEND", "memory")
os.environ.setdefault("LANGSMITH_API_KEY", "")
os.environ.setdefault("LANGSMITH_PROJECT", "test")
os.environ.setdefault("LOG_LEVEL", "ERROR")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://testserver")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100")
os.environ.setdefault("RAG_TOP_K", "3")
os.environ.setdefault("RAG_SCORE_THRESHOLD", "0.0")
os.environ.setdefault("RAG_CHUNK_SIZE", "20")
os.environ.setdefault("RAG_CHUNK_OVERLAP", "5")
os.environ.setdefault("RAG_MAX_DOCUMENT_SIZE", "200000")
os.environ.setdefault("RESOURCE_RAG_MAX_DEFAULT", "1")
os.environ.setdefault("RESOURCE_RAG_MAX_EXPLICIT", "3")
os.environ.setdefault("RESOURCE_RAG_STRONG_MATCH_THRESHOLD", "0.78")
os.environ.setdefault("RESOURCE_RAG_PERMISSION_THRESHOLD", "0.55")
os.environ.setdefault("RESOURCE_RAG_MIN_TURNS_BEFORE_SUGGEST", "3")
os.environ.setdefault("RESOURCE_RAG_MIN_DEPTH_BEFORE_RECOMMEND", "0.75")
os.environ.setdefault("RESOURCE_RAG_COOLDOWN_TURNS", "5")
os.environ["RESOURCE_INTENT_CLASSIFIER"] = "deterministic"
os.environ.setdefault("RESOURCE_INTENT_CLASSIFIER_MODEL", "use_default_llm")
os.environ.setdefault("RESOURCE_INTENT_CONFIDENCE_THRESHOLD", "0.65")
os.environ.setdefault("MEMORY_ENABLED", "false")
os.environ.setdefault("STREAMING_ENABLED", "true")
os.environ.setdefault("ALLOW_WILDCARD_CORS", "false")

from app.core.config import get_settings  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.services.conversation_intelligence import ConversationIntelligence  # noqa: E402


@pytest.fixture(autouse=True)
def database() -> Generator[None, None, None]:
    ConversationIntelligence._contexts.clear()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def settings():
    get_settings.cache_clear()
    return get_settings()

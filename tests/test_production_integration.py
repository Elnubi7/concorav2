import os
import sys
import types

import pytest

from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingProviderError, EmbeddingService
from app.services.llm_service import LLMProviderError, LLMService
from app.services.mbti_service import MbtiService
from app.services.rag_service import RagService
from scripts import smoke_test, validate_env


def test_llm_provider_selection_test(settings):
    style = MbtiService().interpret("INTP")
    service = LLMService(settings.llm_provider, settings.llm_model, settings.effective_llm_api_key, app_env=settings.app_env)
    reply = service.generate_gemma_reply_sync(
        message="مش عارف أنام",
        intent="simple_advice",
        intensity=2,
        style=style,
        rag_context=[],
        plan={},
    )
    assert reply == "test provider reply for simple_advice: مش عارف أنام"


def test_openai_compatible_accepts_dummy_key_in_local_env(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            message = types.SimpleNamespace(content="رد محلي من موديل حقيقي")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    fake_openai = types.SimpleNamespace(
        OpenAI=FakeOpenAI,
        APIConnectionError=Exception,
        APIStatusError=Exception,
        APITimeoutError=Exception,
    )
    monkeypatch.setitem(sys.modules, "open" + "ai", fake_openai)

    style = MbtiService().interpret("INTP")
    service = LLMService(
        "openai_compatible",
        "llama3.1:8b",
        "ollama",
        base_url="http://localhost:11434/v1",
        timeout_seconds=5,
        max_retries=0,
        app_env="local",
    )
    reply = service.generate_gemma_reply_sync(
        message="مش عارف أنام",
        intent="simple_advice",
        intensity=2,
        style=style,
        rag_context=[],
        plan={},
    )

    assert reply == "رد محلي من موديل حقيقي"
    assert captured["client"]["api_key"] == "ollama"
    assert captured["client"]["base_url"] == "http://localhost:11434/v1"


def test_github_models_provider_uses_github_env_values(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            message = types.SimpleNamespace(content="github model reply")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    fake_openai = types.SimpleNamespace(
        OpenAI=FakeOpenAI,
        APIConnectionError=Exception,
        APIStatusError=Exception,
        APITimeoutError=Exception,
    )
    monkeypatch.setitem(sys.modules, "open" + "ai", fake_openai)

    service = LLMService(
        "github_models",
        "openai/gpt" + "-4.1-mini",
        "github-token",
        base_url="https://models.github.ai/inference",
        timeout_seconds=30,
        max_retries=0,
        app_env="local",
    )
    reply = service.generate_simple_reply_sync(message="hello", rag_disabled=False, max_tokens=180, temperature=0.4)

    assert reply == "github model reply"
    assert captured["client"]["api_key"] == "github-token"
    assert captured["client"]["base_url"] == "https://models.github.ai/inference"
    assert captured["request"]["model"] == "openai/gpt" + "-4.1-mini"
    assert captured["request"]["max_tokens"] == 180
    assert captured["request"]["temperature"] == 0.4


def test_invalid_llm_provider_config_fails_controlled(settings):
    style = MbtiService().interpret("INTP")
    service = LLMService("unsupported-provider", settings.llm_model, settings.effective_llm_api_key, app_env=settings.app_env)
    with pytest.raises(LLMProviderError):
        service.generate_gemma_reply_sync(
            message="مرحبا",
            intent="casual_chat",
            intensity=1,
            style=style,
            rag_context=[],
            plan={},
        )


def test_test_llm_provider_rejected_outside_test_env(settings):
    style = MbtiService().interpret("INTP")
    service = LLMService("test", settings.llm_model, settings.effective_llm_api_key, app_env="local")
    with pytest.raises(LLMProviderError, match="APP_ENV=test"):
        service.generate_gemma_reply_sync(
            message="مش عارف أنام",
            intent="simple_advice",
            intensity=2,
            style=style,
            rag_context=[],
            plan={},
        )


def test_embedding_provider_selection_local(settings):
    embedding = EmbeddingService(settings).embed_query("اختبار")
    assert len(embedding) == settings.embedding_dimensions
    assert any(value != 0 for value in embedding)


def test_invalid_embedding_provider_config_fails_controlled(settings, monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "unsupported-provider")
    with pytest.raises(EmbeddingProviderError):
        EmbeddingService(settings).embed_query("اختبار")


def test_document_chunking_uses_configured_overlap(db, settings):
    service = DocumentService(db, settings)
    chunks = service.chunk_text(" ".join(str(index) for index in range(10)), chunk_size=4, chunk_overlap=2)
    assert chunks == ["0 1 2 3", "2 3 4 5", "4 5 6 7", "6 7 8 9", "8 9"]


def test_rag_thresholds_filter_results(db, settings):
    settings.enable_rag = True
    DocumentService(db, settings).upload(user_id="u-threshold", title="doc", content="alpha beta gamma", source=None, metadata={})
    db.commit()

    settings.rag_score_threshold = 1.1
    high_threshold_results = RagService(settings, db).retrieve(query="alpha", user_id="u-threshold", session_id="s1")
    assert high_threshold_results == []

    settings.rag_score_threshold = 0.0
    low_threshold_results = RagService(settings, db).retrieve(query="alpha", user_id="u-threshold", session_id="s1")
    assert low_threshold_results


def test_pgvector_retriever_interface_uses_repository_mock(settings, monkeypatch):
    calls = {}

    class FakeRepo:
        def __init__(self, db):
            calls["db"] = db

        def semantic_search_pgvector(self, query_embedding, user_id, top_k, score_threshold):
            calls["query_embedding"] = query_embedding
            calls["user_id"] = user_id
            calls["top_k"] = top_k
            calls["score_threshold"] = score_threshold
            return [{"chunk_id": "c1", "document_id": "d1", "content": "ctx", "score": 0.9, "metadata": {}}]

        def log_rag_query(self, **kwargs):
            calls["logged"] = kwargs

    monkeypatch.setattr("app.services.rag_service.DocumentRepository", FakeRepo)
    monkeypatch.setattr(settings, "enable_rag", True)
    monkeypatch.setattr(settings, "vector_backend", "pgvector")
    results = RagService(settings, object()).retrieve(query="ctx", user_id="u1", session_id="s1")
    assert results[0]["chunk_id"] == "c1"
    assert calls["logged"]["results"][0]["score"] == 0.9


def test_smoke_test_script_structure():
    assert hasattr(smoke_test, "main")
    assert hasattr(smoke_test, "request_json")
    source = open("scripts/smoke_test.py", encoding="utf-8").read()
    assert "/api/v1/health" in source
    assert "/api/v1/ready" in source
    assert "/api/v1/chat/stream" in source
    assert "BASE_URL" in source


def test_validate_env_fails_bad_production_config(monkeypatch):
    for key in validate_env.REQUIRED_ENV_KEYS:
        monkeypatch.setenv(key, "set")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LLM_PROVIDER", "test")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "test")
    monkeypatch.setenv("JWT_SECRET_KEY", "short")
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    monkeypatch.setenv("VECTOR_BACKEND", "memory")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("ALLOW_WILDCARD_CORS", "false")

    errors = validate_env.validate()
    assert "Production cannot use test LLM provider" in errors
    assert "Production cannot use test embedding provider" in errors
    assert "Production JWT_SECRET_KEY is weak" in errors
    assert "Production cannot use sqlite DATABASE_URL" in errors
    assert "Production cannot use VECTOR_BACKEND=memory" in errors
    assert "Production LLM API key is empty" in errors
    assert "Production wildcard CORS requires ALLOW_WILDCARD_CORS=true" in errors


def test_validate_env_passes_valid_local_config(monkeypatch):
    for key in validate_env.REQUIRED_ENV_KEYS:
        monkeypatch.setenv(key, os.environ.get(key, "set"))
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "ollama")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_API_KEY", "ollama")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    assert validate_env.validate() == []


def test_validate_env_github_models_requires_github_token(monkeypatch):
    for key in validate_env.REQUIRED_ENV_KEYS:
        monkeypatch.setenv(key, os.environ.get(key, "set"))
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("LLM_PROVIDER", "github_models")
    monkeypatch.setenv("GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference")
    monkeypatch.setenv("GITHUB_TOKEN", "")

    assert "GITHUB_TOKEN is required when LLM_PROVIDER is github_models" in validate_env.validate()


def test_validate_env_missing_github_token_allowed_for_openai_compatible(monkeypatch):
    for key in validate_env.REQUIRED_ENV_KEYS:
        monkeypatch.setenv(key, os.environ.get(key, "set"))
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "ollama")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_API_KEY", "ollama")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")

    assert "GITHUB_TOKEN is required when LLM_PROVIDER is github_models" not in validate_env.validate()


def test_validate_env_rejects_test_provider_in_production(monkeypatch):
    for key in validate_env.REQUIRED_ENV_KEYS:
        monkeypatch.setenv(key, os.environ.get(key, "set"))
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LLM_PROVIDER", "test")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("JWT_SECRET_KEY", "x" * 32)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://gemma:gemma@postgres:5432/gemma")
    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
    monkeypatch.setenv("LLM_API_KEY", "real-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "real-key")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")

    assert "Production cannot use test LLM provider" in validate_env.validate()

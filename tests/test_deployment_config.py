from pathlib import Path

import pytest

from app.core.config import RuntimeConfigError, get_settings, runtime_settings_errors, validate_runtime_settings
from app.schemas.chat import ChatRequest, DocumentUploadRequest
from app.services.chat_service import ChatService


def set_production_simple_env(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("APP_NAME", "gemma-chatbot-api")
    monkeypatch.setenv("APP_VERSION", "1.0.0")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LLM_PROVIDER", "github_models")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "x" * 40)
    monkeypatch.setenv("GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt" + "-4.1-mini")
    monkeypatch.setenv("JWT_SECRET_KEY", "s" * 40)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://example.com")
    monkeypatch.setenv("ENABLE_DB", "false")
    monkeypatch.setenv("ENABLE_REDIS", "false")
    monkeypatch.setenv("ENABLE_MEMORY", "false")
    monkeypatch.setenv("ENABLE_RAG", "false")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "disabled")
    monkeypatch.setenv("EMBEDDING_MODEL", "disabled")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "0")
    monkeypatch.setenv("VECTOR_BACKEND", "disabled")
    get_settings.cache_clear()


def test_production_simple_server_config_passes_without_database_redis_or_rag(monkeypatch):
    set_production_simple_env(monkeypatch)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    settings = get_settings()

    assert runtime_settings_errors(settings) == []


def test_production_config_fails_when_github_token_missing(monkeypatch):
    set_production_simple_env(monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "")
    get_settings.cache_clear()

    assert "GITHUB_TOKEN is required when LLM_PROVIDER is github_models" in runtime_settings_errors(get_settings())


def test_production_config_fails_when_jwt_secret_is_placeholder(monkeypatch):
    set_production_simple_env(monkeypatch)
    monkeypatch.setenv("JWT_SECRET_KEY", "local-dev-secret-change-me")
    get_settings.cache_clear()

    assert "JWT_SECRET_KEY must be a strong non-placeholder value in production" in runtime_settings_errors(get_settings())


def test_db_engine_is_not_created_at_import_time(monkeypatch):
    monkeypatch.setenv("ENABLE_DB", "false")
    get_settings.cache_clear()
    import app.db.session as module

    assert "engine" not in module.__dict__
    assert "SessionLocal" not in module.__dict__


def test_get_db_yields_none_when_db_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_DB", "false")
    get_settings.cache_clear()
    import app.db.session as module

    module.reset_db_engine_cache()

    db = next(module.get_db())

    assert db is None


def test_simple_chat_works_with_db_none(settings):
    response = ChatService(settings, None).chat(ChatRequest(user_id="deploy", message="مرحبا", mbti="INTP", metadata={}))

    assert response.path_used == "simple"


def test_document_endpoint_returns_503_when_db_disabled(settings):
    from app.api.v1.routes.documents import upload_document

    settings.enable_db = False
    request = DocumentUploadRequest(title="doc", content="content", metadata={})

    with pytest.raises(Exception) as exc:
        upload_document(request, db=None, settings=settings)

    assert getattr(exc.value, "status_code", None) == 503
    assert getattr(exc.value, "message", "") == "Database is disabled"


def test_startup_validation_returns_clear_errors(monkeypatch):
    set_production_simple_env(monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "")
    get_settings.cache_clear()

    with pytest.raises(RuntimeConfigError) as exc:
        validate_runtime_settings(get_settings())

    assert "GITHUB_TOKEN is required when LLM_PROVIDER is github_models" in exc.value.errors


def test_env_server_example_contains_placeholders_only():
    text = Path(".env.server.example").read_text(encoding="utf-8")

    assert "GITHUB_TOKEN=" in text
    assert "JWT_SECRET_KEY=" in text
    assert "ghp_" not in text
    assert "github_pat_" not in text
    assert "sk" + "-" not in text


def test_env_examples_do_not_contain_real_secrets():
    for path in (Path(".env.example"), Path(".env.server.example")):
        text = path.read_text(encoding="utf-8")
        assert "github_pat_" not in text
        assert "ghp_" not in text
        assert "sk" + "-" not in text

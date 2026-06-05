import os
import sys

REQUIRED_ENV_KEYS = {
    "APP_ENV",
    "APP_NAME",
    "APP_VERSION",
    "API_HOST",
    "API_PORT",
    "DATABASE_URL",
    "REDIS_URL",
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_MODELS_BASE_URL",
    "LLM_API_KEY",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "LLM_MAX_TOKENS",
    "LLM_TEMPERATURE",
    "SIMPLE_CHAT_MODE",
    "HYBRID_ROUTER_MODE",
    "COMPLEXITY_THRESHOLD",
    "ENABLE_RAG",
    "ENABLE_RESOURCE_RAG",
    "ENABLE_MEMORY",
    "ENABLE_LANGGRAPH",
    "ENABLE_DB",
    "ENABLE_REDIS",
    "ALWAYS_FAST_CHAT",
    "LLM_FAST_MAX_TOKENS",
    "LLM_FAST_TEMPERATURE",
    "LLM_FAST_TIMEOUT_SECONDS",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_DIMENSIONS",
    "VECTOR_BACKEND",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LOG_LEVEL",
    "JWT_SECRET_KEY",
    "CORS_ALLOWED_ORIGINS",
    "RATE_LIMIT_ENABLED",
    "RATE_LIMIT_PER_MINUTE",
    "RAG_TOP_K",
    "RAG_SCORE_THRESHOLD",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_MAX_DOCUMENT_SIZE",
    "RESOURCE_RAG_MAX_DEFAULT",
    "RESOURCE_RAG_MAX_EXPLICIT",
    "RESOURCE_RAG_STRONG_MATCH_THRESHOLD",
    "RESOURCE_RAG_PERMISSION_THRESHOLD",
    "RESOURCE_RAG_MIN_TURNS_BEFORE_SUGGEST",
    "RESOURCE_RAG_MIN_DEPTH_BEFORE_RECOMMEND",
    "RESOURCE_RAG_COOLDOWN_TURNS",
    "RESOURCE_INTENT_CLASSIFIER",
    "RESOURCE_INTENT_CLASSIFIER_MODEL",
    "RESOURCE_INTENT_CONFIDENCE_THRESHOLD",
    "MEMORY_ENABLED",
    "STREAMING_ENABLED",
    "ALLOW_WILDCARD_CORS",
}

TEST_PROVIDER = "test"
SUPPORTED_LLM_PROVIDERS = {TEST_PROVIDER, "openai_compatible", "github_models"}
SUPPORTED_EMBEDDING_PROVIDERS = {TEST_PROVIDER, "openai", "openai_compatible"}
PLACEHOLDER_API_KEYS = {"", "local", "ollama", "lmstudio", "lm-studio", "dummy", "test", "changeme", "change-me"}
WEAK_SECRET_VALUES = {"", "secret", "test-secret", "local-dev-secret", "replace-with-local-secret"}


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


def is_true(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def validate() -> list[str]:
    errors: list[str] = []
    missing = sorted(key for key in REQUIRED_ENV_KEYS if key not in os.environ)
    if missing:
        errors.append("Missing required env vars: " + ", ".join(missing))

    app_env = os.environ.get("APP_ENV", "").lower()
    is_production = app_env == "production"
    llm_provider = os.environ.get("LLM_PROVIDER", "").lower()
    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
    llm_key = os.environ.get("GITHUB_TOKEN") if llm_provider == "github_models" else os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    embedding_key = os.environ.get("EMBEDDING_API_KEY") or llm_key

    if llm_provider not in SUPPORTED_LLM_PROVIDERS:
        errors.append("Unsupported LLM_PROVIDER; use test, openai_compatible, or github_models")
    if embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        errors.append("Unsupported EMBEDDING_PROVIDER; use test, openai, or openai_compatible")
    if llm_provider == "openai_compatible" and not os.environ.get("LLM_BASE_URL"):
        errors.append("LLM_BASE_URL is required when LLM_PROVIDER is openai_compatible")
    if llm_provider == "github_models":
        if not os.environ.get("GITHUB_MODELS_BASE_URL"):
            errors.append("GITHUB_MODELS_BASE_URL is required when LLM_PROVIDER is github_models")
        if not os.environ.get("GITHUB_TOKEN"):
            errors.append("GITHUB_TOKEN is required when LLM_PROVIDER is github_models")
    if embedding_provider == "openai_compatible" and not os.environ.get("EMBEDDING_BASE_URL"):
        errors.append("EMBEDDING_BASE_URL is required for EMBEDDING_PROVIDER=openai_compatible")
    if not is_production and app_env != "test" and llm_provider == TEST_PROVIDER:
        errors.append("LLM_PROVIDER test mode is only allowed when APP_ENV is test")
    if not is_production and app_env != "test" and embedding_provider == TEST_PROVIDER:
        errors.append("EMBEDDING_PROVIDER test mode is only allowed when APP_ENV is test")

    if not is_production:
        return errors

    allow_placeholder_keys = is_true(os.environ.get("ALLOW_PLACEHOLDER_LOCAL_API_KEYS"))
    if llm_provider == TEST_PROVIDER:
        errors.append("Production cannot use test LLM provider")
    if embedding_provider == TEST_PROVIDER:
        errors.append("Production cannot use test embedding provider")
    if os.environ.get("JWT_SECRET_KEY", "") in WEAK_SECRET_VALUES or len(os.environ.get("JWT_SECRET_KEY", "")) < 32:
        errors.append("Production JWT_SECRET_KEY is weak")
    if os.environ.get("DATABASE_URL", "").startswith("sqlite"):
        errors.append("Production cannot use sqlite DATABASE_URL")
    if os.environ.get("VECTOR_BACKEND", "").lower() == "memory":
        errors.append("Production cannot use VECTOR_BACKEND=memory")
    if not llm_key:
        errors.append("Production LLM API key is empty")
    if llm_key.lower() in PLACEHOLDER_API_KEYS and not allow_placeholder_keys:
        errors.append("Production LLM API key cannot be a placeholder")
    if embedding_provider in {"openai", "openai_compatible"} and embedding_key.lower() in PLACEHOLDER_API_KEYS and not allow_placeholder_keys:
        errors.append("Production embedding API key cannot be a placeholder")
    if "*" in [origin.strip() for origin in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",")] and not is_true(
        os.environ.get("ALLOW_WILDCARD_CORS")
    ):
        errors.append("Production wildcard CORS requires ALLOW_WILDCARD_CORS=true")
    return errors


def main() -> int:
    load_dotenv()
    errors = validate()
    if errors:
        print("FAIL environment validation")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS environment validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())

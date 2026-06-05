from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(..., alias="APP_ENV")
    app_name: str = Field(..., alias="APP_NAME")
    app_version: str = Field(..., alias="APP_VERSION")
    api_host: str = Field(..., alias="API_HOST")
    api_port: int = Field(..., alias="API_PORT")
    database_url: str = Field(..., alias="DATABASE_URL")
    redis_url: str = Field(..., alias="REDIS_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    github_models_base_url: str | None = Field(default=None, alias="GITHUB_MODELS_BASE_URL")
    llm_provider: str = Field(..., alias="LLM_PROVIDER")
    llm_model: str = Field(..., alias="LLM_MODEL")
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")
    llm_timeout_seconds: float = Field(..., alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(..., alias="LLM_MAX_RETRIES")
    llm_max_tokens: int = Field(default=120, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.4, alias="LLM_TEMPERATURE")
    simple_chat_mode: bool = Field(default=True, alias="SIMPLE_CHAT_MODE")
    hybrid_router_mode: Literal["deterministic"] = Field(default="deterministic", alias="HYBRID_ROUTER_MODE")
    complexity_threshold: float = Field(default=0.65, alias="COMPLEXITY_THRESHOLD")
    enable_rag: bool = Field(default=False, alias="ENABLE_RAG")
    enable_resource_rag: bool = Field(default=True, alias="ENABLE_RESOURCE_RAG")
    enable_langgraph: bool = Field(default=False, alias="ENABLE_LANGGRAPH")
    enable_memory: bool = Field(default=False, validation_alias=AliasChoices("ENABLE_MEMORY", "MEMORY_ENABLED"))
    enable_db: bool = Field(default=False, alias="ENABLE_DB")
    enable_redis: bool = Field(default=False, alias="ENABLE_REDIS")
    always_fast_chat: bool = Field(default=False, alias="ALWAYS_FAST_CHAT")
    llm_fast_max_tokens: int = Field(default=120, alias="LLM_FAST_MAX_TOKENS")
    llm_fast_temperature: float = Field(default=0.4, alias="LLM_FAST_TEMPERATURE")
    llm_fast_timeout_seconds: float = Field(default=30, alias="LLM_FAST_TIMEOUT_SECONDS")
    embedding_provider: str = Field(..., alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(..., alias="EMBEDDING_MODEL")
    embedding_api_key: str | None = Field(default=None, alias="EMBEDDING_API_KEY")
    embedding_base_url: str | None = Field(default=None, alias="EMBEDDING_BASE_URL")
    embedding_dimensions: int = Field(..., alias="EMBEDDING_DIMENSIONS")
    vector_backend: Literal["pgvector", "qdrant", "memory"] = Field(..., alias="VECTOR_BACKEND")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str | None = Field(default=None, alias="LANGSMITH_PROJECT")
    log_level: str = Field(..., alias="LOG_LEVEL")
    jwt_secret_key: str = Field(..., alias="JWT_SECRET_KEY")
    cors_allowed_origins: str = Field(..., alias="CORS_ALLOWED_ORIGINS")
    rate_limit_enabled: bool = Field(..., alias="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(..., alias="RATE_LIMIT_PER_MINUTE")
    rag_top_k: int = Field(..., alias="RAG_TOP_K")
    rag_score_threshold: float = Field(..., alias="RAG_SCORE_THRESHOLD")
    rag_chunk_size: int = Field(..., alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(..., alias="RAG_CHUNK_OVERLAP")
    rag_max_document_size: int = Field(..., alias="RAG_MAX_DOCUMENT_SIZE")
    resource_rag_max_default: int = Field(default=1, alias="RESOURCE_RAG_MAX_DEFAULT")
    resource_rag_max_explicit: int = Field(default=3, alias="RESOURCE_RAG_MAX_EXPLICIT")
    resource_rag_strong_match_threshold: float = Field(default=0.78, alias="RESOURCE_RAG_STRONG_MATCH_THRESHOLD")
    resource_rag_permission_threshold: float = Field(default=0.55, alias="RESOURCE_RAG_PERMISSION_THRESHOLD")
    resource_rag_min_turns_before_suggest: int = Field(default=3, alias="RESOURCE_RAG_MIN_TURNS_BEFORE_SUGGEST")
    resource_rag_min_depth_before_recommend: float = Field(default=0.75, alias="RESOURCE_RAG_MIN_DEPTH_BEFORE_RECOMMEND")
    resource_rag_cooldown_turns: int = Field(default=5, alias="RESOURCE_RAG_COOLDOWN_TURNS")
    resource_intent_classifier: Literal["semantic", "deterministic"] = Field(default="semantic", alias="RESOURCE_INTENT_CLASSIFIER")
    resource_intent_classifier_model: str = Field(default="use_default_llm", alias="RESOURCE_INTENT_CLASSIFIER_MODEL")
    resource_intent_confidence_threshold: float = Field(default=0.65, alias="RESOURCE_INTENT_CONFIDENCE_THRESHOLD")
    streaming_enabled: bool = Field(..., alias="STREAMING_ENABLED")
    allow_wildcard_cors: bool = Field(default=False, alias="ALLOW_WILDCARD_CORS")
    archive_deleted_sessions: bool = Field(default=True, alias="ARCHIVE_DELETED_SESSIONS")
    metrics_token: str | None = Field(default=None, alias="METRICS_TOKEN")

    @property
    def effective_llm_api_key(self) -> str | None:
        if self.llm_provider == "github_models":
            return self.github_token
        return self.llm_api_key or self.openai_api_key

    @property
    def effective_llm_base_url(self) -> str | None:
        if self.llm_provider == "github_models":
            return self.github_models_base_url
        return self.llm_base_url

    @property
    def effective_embedding_api_key(self) -> str | None:
        return self.embedding_api_key or self.effective_llm_api_key

    @property
    def memory_enabled(self) -> bool:
        return self.enable_memory

    @property
    def cors_origins(self) -> list[str]:
        raw = self.cors_allowed_origins.strip()
        if raw.startswith("["):
            import json

            parsed = json.loads(raw)
            return [str(item) for item in parsed]
        return [item.strip() for item in raw.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

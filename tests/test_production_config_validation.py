import ast
from pathlib import Path

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

FORBIDDEN_CODE_MARKERS = (
    "sk" + "-",
    "hardcoded" + "-api-key",
    "postgresql://" + "user:password",
    "gpt" + "-3",
    "gpt" + "-4",
    "gpt" + "-5",
    "DATABASE" + "_URL=",
    "LLM" + "_PROVIDER=",
)

PROVIDER_VALUES = ("open" + "ai", "anth" + "ropic", "qd" + "rant")
ALLOWED_PROVIDER_FILES = {
    Path("app/core/config.py"),
    Path("app/services/embedding_service.py"),
    Path("app/services/llm_service.py"),
    Path("scripts/validate_env.py"),
    Path("requirements.txt"),
    Path(".env.example"),
    Path("README.md"),
}


def env_example_keys() -> set[str]:
    keys: set[str] = set()
    for line in Path(".env.example").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            keys.add(line.split("=", 1)[0])
    return keys


def test_env_example_contains_all_required_keys():
    assert REQUIRED_ENV_KEYS <= env_example_keys()


def test_no_hardcoded_secrets_models_urls_or_provider_settings_in_python_code():
    for path in Path(".").rglob("*.py"):
        if any(part in {".venv", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in FORBIDDEN_CODE_MARKERS), path
        rel_path = path.relative_to(Path("."))
        if rel_path not in ALLOWED_PROVIDER_FILES:
            assert not any(f'"{value}"' in text or f"'{value}'" in text for value in PROVIDER_VALUES), path


def test_prompts_are_loaded_from_prompt_files_not_buried_in_python():
    prompt_dir = Path("app/prompts/v1")
    expected_prompts = {
        "gemma_persona.md",
        "mbti_interpreter.md",
        "intent_router.md",
        "problem_detector.md",
        "response_planner.md",
        "quality_guard.md",
        "safety_policy.md",
        "rag_answering.md",
        "clarification_response.md",
        "gemma_fast_chat.md",
        "gemma_simple_chat.md",
        "gemma_langgraph_reasoning.md",
        "gemma_resource_permission.md",
        "gemma_resource_recommendation.md",
        "resource_intent_classifier.md",
    }
    assert expected_prompts <= {path.name for path in prompt_dir.glob("*.md")}

    loader_source = Path("app/prompts/loader.py").read_text(encoding="utf-8")
    assert "read_text" in loader_source

    for path in Path("app").rglob("*.py"):
        if path.parts[:2] == ("app", "prompts"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        large_arabic_literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) > 700
            and any("\u0600" <= char <= "\u06ff" for char in node.value)
        ]
        assert not large_arabic_literals, path


def test_fast_prompt_is_compact_for_local_ollama():
    prompt = Path("app/prompts/v1/gemma_fast_chat.md").read_text(encoding="utf-8")
    assert len(prompt.split()) <= 120
    assert "MBTI" in prompt
    assert "سؤال واحد" in prompt


def test_llm_service_has_no_canned_normal_operation_replies():
    source = Path("app/services/llm_service.py").read_text(encoding="utf-8")
    assert "_generate_local_reply" not in source
    assert ("LLM" + "_PROVIDER=" + "local") not in source
    assert "Deterministic test-only provider" in source
    assert "openai_compatible" in source

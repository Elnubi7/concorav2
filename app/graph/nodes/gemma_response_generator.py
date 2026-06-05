from app.core.errors import AppError
from app.graph.state import GraphState
from app.services.llm_service import LLMProviderError, LLMService


def gemma_response_generator(state: GraphState) -> GraphState:
    settings = state["settings"]
    llm = LLMService(
        settings.llm_provider,
        settings.llm_model,
        settings.effective_llm_api_key,
        base_url=settings.effective_llm_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        app_env=settings.app_env,
    )
    try:
        state["reply"] = llm.generate_gemma_reply_sync(
            message=state["message"],
            intent=state["intent"],
            intensity=state["intensity"],
            style=state["mbti_style"],
            rag_context=state.get("rag_context", []),
            plan=state["response_plan"],
        )
    except LLMProviderError as exc:
        raise AppError("LLM provider unavailable", 503) from exc
    return state

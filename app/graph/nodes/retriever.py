from app.core.errors import AppError
from app.graph.state import GraphState
from app.services.embedding_service import EmbeddingProviderError
from app.services.metrics_service import MetricsService
from app.services.rag_service import RagService


def retriever(state: GraphState) -> GraphState:
    service = RagService(state["settings"], state["db"])
    try:
        state["rag_context"] = service.retrieve(query=state["message"], user_id=state["user_id"], session_id=state["session_id"])
    except EmbeddingProviderError as exc:
        raise AppError("Embedding provider unavailable", 503) from exc
    except NotImplementedError as exc:
        raise AppError("Vector backend unavailable", 503) from exc
    state["used_rag"] = True
    MetricsService.increment_rag_query()
    return state

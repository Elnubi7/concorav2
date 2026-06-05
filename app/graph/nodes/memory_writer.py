from uuid import uuid4

from app.db.repositories.chat_repository import ChatRepository
from app.graph.state import GraphState
from app.services.memory_service import MemoryService
from app.services.metrics_service import MetricsService


def memory_writer(state: GraphState) -> GraphState:
    if state.get("db") is None:
        state["assistant_message_id"] = f"graph-{uuid4()}"
        MetricsService.increment_graph_run()
        return state
    repo = ChatRepository(state["db"])
    assistant_message = repo.add_message(
        user_id=state["user_id"],
        session_id=state["session_id"],
        role="assistant",
        content=state["reply"],
        intent=state["intent"],
        intensity=state["intensity"],
        used_rag=state.get("used_rag", False),
        mbti=state.get("mbti"),
        metadata={"quality_issues": state.get("quality_issues", [])},
    )
    state["assistant_message_id"] = assistant_message.id
    MemoryService(state["settings"], state["db"]).maybe_write(state["user_id"], state["message"])
    repo.add_graph_run(
        user_id=state["user_id"],
        session_id=state["session_id"],
        request_id=state.get("request_id"),
        final_intent=state["intent"],
        intensity=state["intensity"],
        used_rag=state.get("used_rag", False),
        node_latencies_ms=state.get("node_latencies_ms", {}),
        metadata={"intent_confidence": state.get("intent_confidence"), "quality_issues": state.get("quality_issues", [])},
    )
    MetricsService.increment_graph_run()
    state["db"].commit()
    return state

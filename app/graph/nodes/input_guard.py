from app.core.errors import AppError
from app.graph.state import GraphState


def input_guard(state: GraphState) -> GraphState:
    if not state.get("message", "").strip():
        raise AppError("Message is required", 422)
    if len(state["message"]) > 12000:
        raise AppError("Message is too long", 413)
    return state

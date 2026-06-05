from app.graph.state import GraphState
from app.services.quality_service import QualityService


def quality_guard(state: GraphState) -> GraphState:
    service = QualityService()
    ok, issues = service.check(
        reply=state["reply"],
        intent=state["intent"],
        used_rag=state.get("used_rag", False),
        rag_required=state.get("rag_required", False),
        rag_context=state.get("rag_context", []),
    )
    state["quality_issues"] = issues
    _ = ok
    return state

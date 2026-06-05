from app.graph.state import GraphState
from app.services.rag_service import RagService


def rag_decider(state: GraphState) -> GraphState:
    service = RagService(state["settings"], state["db"])
    state["rag_required"] = service.should_retrieve(intent=state["intent"], message=state["message"])
    state["used_rag"] = False
    return state

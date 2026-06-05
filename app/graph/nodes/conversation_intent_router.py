from app.graph.state import GraphState
from app.services.intent_service import IntentService


def conversation_intent_router(state: GraphState) -> GraphState:
    result = IntentService().classify(state["message"])
    state["intent"] = result.intent
    state["intent_confidence"] = result.confidence
    state["intent_reasoning"] = result.reasoning
    return state

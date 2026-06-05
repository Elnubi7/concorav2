from app.graph.state import GraphState
from app.services.intent_service import IntensityService


def response_intensity_classifier(state: GraphState) -> GraphState:
    state["intensity"] = IntensityService().classify(state["intent"], state["message"])
    return state

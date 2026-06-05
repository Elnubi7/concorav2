from app.graph.state import GraphState
from app.services.mbti_service import MbtiService


def mbti_interpreter(state: GraphState) -> GraphState:
    state["mbti_style"] = MbtiService().interpret(state["mbti"])
    return state

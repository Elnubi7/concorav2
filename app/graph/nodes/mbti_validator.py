from app.core.errors import AppError
from app.graph.state import GraphState
from app.services.mbti_service import MbtiService


def mbti_validator(state: GraphState) -> GraphState:
    try:
        state["mbti"] = MbtiService().validate(state["mbti"])
    except ValueError as exc:
        raise AppError(str(exc), 422) from exc
    return state

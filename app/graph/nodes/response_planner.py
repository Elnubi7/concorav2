from app.graph.state import GraphState


def response_planner(state: GraphState) -> GraphState:
    style = state["mbti_style"]
    state["response_plan"] = {
        "mode": state["conversation_mode"],
        "tone": style.tone,
        "challenge_level": style.challenge_level,
        "answer_length": style.answer_length,
        "structure_preference": style.structure_preference,
        "avoid_patterns": style.avoid_patterns,
        "one_question_max": True,
    }
    return state

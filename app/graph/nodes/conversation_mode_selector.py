from app.graph.state import GraphState


def conversation_mode_selector(state: GraphState) -> GraphState:
    mapping = {
        "casual_chat": "normal",
        "simple_advice": "practical",
        "emotional_support": "supportive",
        "deep_problem": "coaching",
        "decision_problem": "decision",
        "knowledge_question": "knowledge",
        "needs_rag": "rag",
        "clarification_needed": "clarification",
        "crisis_or_safety": "safety",
    }
    state["conversation_mode"] = mapping[state["intent"]]
    return state

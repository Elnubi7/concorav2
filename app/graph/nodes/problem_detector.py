from app.graph.state import GraphState


def problem_detector(state: GraphState) -> GraphState:
    if state["intent"] == "clarification_needed":
        state["problem_profile"] = {"repeated": False, "rumination": False, "needs_coaching": False}
        return state
    text = state["message"].lower()
    repeated = any(term in text for term in ("بقالي", "شهر", "أسابيع", "كل يوم", "دايما"))
    rumination = any(term in text for term in ("بفضل أفكر", "تفكير", "مش بعرف أوقف"))
    state["problem_profile"] = {"repeated": repeated, "rumination": rumination, "needs_coaching": repeated or rumination}
    if state["intent"] == "emotional_support" and repeated:
        state["intent"] = "deep_problem"
        state["intensity"] = max(state.get("intensity", 3), 4)
    return state

from typing import Any, TypedDict

from app.schemas.chat import Intent
from app.services.mbti_service import MbtiStyle


class GraphState(TypedDict, total=False):
    user_id: str
    session_id: str | None
    message: str
    mbti: str
    metadata: dict[str, Any]
    request_id: str | None
    db: Any
    settings: Any
    session_messages: list[dict[str, Any]]
    memory: dict[str, Any]
    mbti_style: MbtiStyle
    intent: Intent
    intent_confidence: float
    intent_reasoning: str
    intensity: int
    problem_profile: dict[str, Any]
    conversation_mode: str
    rag_required: bool
    rag_context: list[dict[str, Any]]
    response_plan: dict[str, Any]
    reply: str
    used_rag: bool
    user_message_id: str
    assistant_message_id: str
    quality_issues: list[str]
    node_latencies_ms: dict[str, float]

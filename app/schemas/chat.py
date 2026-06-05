from typing import Any, Literal

from pydantic import BaseModel, Field

Intent = Literal[
    "casual_chat",
    "simple_advice",
    "emotional_support",
    "deep_problem",
    "decision_problem",
    "knowledge_question",
    "needs_rag",
    "clarification_needed",
    "crisis_or_safety",
]
ResourceAction = Literal["recommend_now", "ask_permission", "no_recommendation"]


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str | None = None
    message: str = Field(min_length=1)
    mbti: str = Field(min_length=4, max_length=4)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    intent: Intent
    intensity: int = Field(ge=1, le=5)
    used_rag: bool
    used_resource_rag: bool = False
    resource_action: ResourceAction = "no_recommendation"
    recommended_resources: list[dict[str, Any]] = Field(default_factory=list)
    conversation_stage: str | None = None
    depth_score: float | None = None
    complexity_score: float | None = None
    resource_reason: str | None = None
    path_used: Literal["simple", "fast", "full_graph", "langgraph", "resource_rag", "rag"] = "full_graph"
    model: str | None = None
    latency_ms: int | None = None
    route_reason: str | None = None
    session_id: str
    message_id: str


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    intent: str | None = None
    intensity: int | None = None
    used_rag: bool = False
    created_at: str


class SessionOut(BaseModel):
    id: str
    user_id: str
    is_archived: bool
    messages: list[MessageOut]


class DocumentUploadRequest(BaseModel):
    user_id: str | None = None
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    document_id: str
    chunks_indexed: int


class ReindexResponse(BaseModel):
    documents_seen: int
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    app_name: str
    app_version: str


class ReadyResponse(BaseModel):
    status: str
    database: str
    redis: str


class MetricsResponse(BaseModel):
    requests_total: int
    graph_runs_total: int
    rag_queries_total: int

from dataclasses import dataclass, field
from typing import Literal

from app.core.config import Settings
from app.core.policy_loader import load_policy
from app.services.intent_service import IntentService

ConversationStage = Literal["casual", "initial_problem", "deepening", "action_planning", "resource_ready"]
UserNeed = Literal[
    "chat",
    "clarification",
    "simple_advice",
    "emotional_support",
    "analysis",
    "decision_help",
    "resource_request",
    "document_request",
]


@dataclass
class SessionIssueContext:
    current_issue_tags: list[str] = field(default_factory=list)
    issue_thread_turn_count: int = 0
    depth_score: float = 0.0
    turn_index: int = 0
    last_recommended_resource_ids: list[str] = field(default_factory=list)
    last_resource_recommended_at_turn: int = -1000
    user_declined_resources: bool = False


@dataclass(frozen=True)
class ConversationIntelligenceResult:
    conversation_stage: ConversationStage
    user_need: UserNeed
    depth_score: float
    complexity_score: float
    issue_tags: list[str]
    stable_issue_detected: bool
    user_requested_resource: bool
    user_requested_document: bool
    should_use_langgraph: bool
    should_consider_resource: bool
    route_reason: str
    context: SessionIssueContext


class ConversationIntelligence:
    _contexts: dict[str, SessionIssueContext] = {}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.intent_service = IntentService()
        self.conversation_policy = load_policy("conversation_policy")["deterministic_patterns"]
        self.resource_policy = load_policy("resource_rag_policy")["deterministic_patterns"]

    def analyze(self, *, user_id: str, session_id: str | None, message: str, metadata: dict) -> ConversationIntelligenceResult:
        text = message.strip().lower()
        context = self._context(user_id, session_id)
        context.turn_index += 1

        user_requested_document = any(term.lower() in text for term in self.conversation_policy["document_request"])
        user_requested_resource = False
        if any(term.lower() in text for term in self.resource_policy["decline_resources"]):
            context.user_declined_resources = True

        intent = self.intent_service.classify(message).intent
        issue_tags = self._issue_tags(text, intent)
        issue_changed = self._issue_changed(context.current_issue_tags, issue_tags)
        if issue_changed:
            context.issue_thread_turn_count = 0
            context.last_resource_recommended_at_turn = -1000
            context.user_declined_resources = False
        if issue_tags:
            context.issue_thread_turn_count += 1
            context.current_issue_tags = issue_tags

        depth_score = self._depth_score(text, issue_tags, context.issue_thread_turn_count)
        complexity_score = self._complexity_score(text, intent, depth_score)
        deep_signal = any(term.lower() in text for term in self.conversation_policy["deep"])
        context.depth_score = max(context.depth_score if not issue_changed else 0.0, depth_score)

        user_need = self._user_need(intent, user_requested_resource, user_requested_document, text)
        stable_issue_detected = bool(issue_tags) and context.issue_thread_turn_count >= self.settings.resource_rag_min_turns_before_suggest
        should_use_langgraph = self.settings.enable_langgraph and (
            user_need in {"analysis", "decision_help"}
            or complexity_score >= self.settings.complexity_threshold
            or (deep_signal and depth_score >= 0.65)
            or depth_score >= self.settings.resource_rag_min_depth_before_recommend
        )
        should_consider_resource = self.settings.enable_resource_rag and not user_requested_document and (
            user_requested_resource or (stable_issue_detected and depth_score >= self.settings.resource_rag_permission_threshold)
        )
        stage = self._stage(user_need, depth_score, context.issue_thread_turn_count, should_consider_resource)
        reason = self._route_reason(user_requested_document, user_requested_resource, should_use_langgraph, should_consider_resource)
        return ConversationIntelligenceResult(
            conversation_stage=stage,
            user_need=user_need,
            depth_score=round(depth_score, 3),
            complexity_score=round(complexity_score, 3),
            issue_tags=issue_tags,
            stable_issue_detected=stable_issue_detected,
            user_requested_resource=user_requested_resource,
            user_requested_document=user_requested_document,
            should_use_langgraph=should_use_langgraph,
            should_consider_resource=should_consider_resource,
            route_reason=reason,
            context=context,
        )

    def _context(self, user_id: str, session_id: str | None) -> SessionIssueContext:
        key = session_id or user_id
        self._contexts.setdefault(key, SessionIssueContext())
        return self._contexts[key]

    def _issue_tags(self, text: str, intent: str) -> list[str]:
        tags = [
            tag
            for tag, keywords in self.conversation_policy["issues"].items()
            if any(keyword.lower() in text for keyword in keywords)
        ]
        if intent == "decision_problem" and "decision" not in tags:
            tags.append("decision")
        return tags

    def _issue_changed(self, previous: list[str], current: list[str]) -> bool:
        return bool(previous and current and not set(previous).intersection(current))

    def _depth_score(self, text: str, issue_tags: list[str], issue_turns: int) -> float:
        score = 0.0
        if issue_tags:
            score += 0.25
        if any(term.lower() in text for term in self.conversation_policy["deep"]):
            score += 0.35
        score += min(0.3, issue_turns * 0.1)
        if len(text.split()) >= 35:
            score += 0.2
        return min(score, 1.0)

    def _complexity_score(self, text: str, intent: str, depth_score: float) -> float:
        score = depth_score * 0.45
        if any(term.lower() in text for term in self.conversation_policy["analysis"]):
            score += 0.55
        if intent == "decision_problem":
            score += 0.3
        score += min(0.3, 0.07 * sum(1 for term in self.conversation_policy["constraints"] if term.lower() in text))
        if len(text.split()) >= 45:
            score += 0.25
        return min(score, 1.0)

    def _user_need(self, intent: str, resource: bool, document: bool, text: str) -> UserNeed:
        if document:
            return "document_request"
        if resource:
            return "resource_request"
        if intent == "clarification_needed":
            return "clarification"
        if intent == "simple_advice":
            return "simple_advice"
        if intent == "emotional_support":
            return "emotional_support"
        if intent == "decision_problem":
            return "decision_help"
        if any(term.lower() in text for term in self.conversation_policy["analysis"]):
            return "analysis"
        return "chat"

    def _stage(self, user_need: UserNeed, depth_score: float, turns: int, resource: bool) -> ConversationStage:
        if resource and depth_score >= self.settings.resource_rag_min_depth_before_recommend:
            return "resource_ready"
        if user_need in {"analysis", "decision_help"}:
            return "action_planning"
        if turns >= 2 or depth_score >= 0.55:
            return "deepening"
        if user_need in {"simple_advice", "emotional_support"}:
            return "initial_problem"
        return "casual"

    def _route_reason(self, document: bool, resource: bool, graph: bool, consider_resource: bool) -> str:
        if document:
            return "document_request"
        if resource:
            return "explicit_resource_request"
        if consider_resource:
            return "resource_candidate"
        if graph:
            return "complex_or_deep_issue"
        return "simple_default"

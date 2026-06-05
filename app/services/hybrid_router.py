from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings
from app.services.intent_service import IntentService

HybridRoute = Literal["simple", "langgraph", "rag_disabled", "rag"]


@dataclass(frozen=True)
class HybridRouteDecision:
    route: HybridRoute
    complexity_score: float
    intent_hint: str
    route_reason: str


class HybridRouter:
    rag_terms = (
        "file",
        "document",
        "uploaded",
        "knowledge base",
        "ملف",
        "الملف",
        "مستند",
        "المستند",
        "المرجع",
        "الداتا",
        "اللي رفعته",
        "ارجعي للملف",
        "بناءً على الملف",
        "بناء على الملف",
        "حسب المستند",
    )
    analysis_terms = (
        "analyze",
        "analysis",
        "plan",
        "step by step",
        "strategy",
        "حللي",
        "تحليل",
        "خطة",
        "خطوة خطوة",
        "استراتيجية",
        "بالتفصيل",
    )
    multiple_goal_terms = ("وفي نفس الوقت", "بس كمان", "مع إن", "لكن", "constraints", "tradeoff")
    constraint_terms = ("لو", "عشان", "ميزانية", "وقت", "أهل", "شغل", "دراسة", "فلوس", "مسؤولية")
    repeated_problem_terms = ("بقالي", "شهر", "أسابيع", "اسابيع", "كل يوم", "دايما", "بفضل أفكر")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.intent_service = IntentService()

    def route(self, message: str) -> HybridRouteDecision:
        text = message.strip().lower()
        intent = self.intent_service.classify(message).intent
        if self._explicit_rag(text):
            route: HybridRoute = "rag" if self.settings.enable_rag else "rag_disabled"
            return HybridRouteDecision(route, 0.2, intent, f"{route}:explicit_document_reference")

        score = self._complexity_score(text, intent)
        if self.settings.enable_langgraph and score >= self.settings.complexity_threshold:
            return HybridRouteDecision("langgraph", score, intent, "langgraph:complex_message")
        return HybridRouteDecision("simple", score, intent, "simple:fast_default")

    def _explicit_rag(self, text: str) -> bool:
        return any(term in text for term in self.rag_terms)

    def _complexity_score(self, text: str, intent: str) -> float:
        words = text.split()
        score = 0.0
        has_analysis_signal = any(term in text for term in self.analysis_terms)
        if has_analysis_signal:
            score += 0.7
        if intent == "decision_problem":
            score += 0.35
            score += min(0.35, 0.08 * sum(1 for term in self.constraint_terms if term in text))
        if intent == "deep_problem":
            score += 0.25
        if any(term in text for term in self.repeated_problem_terms):
            score += 0.45
        if any(term in text for term in self.multiple_goal_terms):
            score += 0.25
        if len(words) >= 45:
            score += 0.35
        elif len(words) >= 28:
            score += 0.2
        if (
            not has_analysis_signal
            and intent in {"clarification_needed", "simple_advice", "knowledge_question", "casual_chat"}
            and len(words) < 28
        ):
            score = min(score, 0.45)
        return min(score, 1.0)

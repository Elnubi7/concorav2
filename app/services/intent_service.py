from dataclasses import dataclass

from app.schemas.chat import Intent


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    confidence: float
    reasoning: str


class IntentService:
    crisis_terms = ("انتحار", "هموت نفسي", "أؤذي نفسي", "اقتل", "مش عايز أعيش", "self harm", "suicide")
    rag_terms = (
        "الملف",
        "المستند",
        "الوثيقة",
        "الداتا",
        "ارجعي لـ",
        "ارجعي ل",
        "بناءً على اللي رفعته",
        "بناء على اللي رفعته",
        "حسب المرجع",
        "knowledge base",
        "document",
        "uploaded file",
        "reference",
    )
    decision_terms = ("اختار", "أقرر", "قرار", "أعمل إيه", "اعمل ايه", "محتار", "محتارة", "أسيب", "اسيب", "أكمل", "اكمل")
    deep_terms = ("بقالي", "شهر", "أسابيع", "كل يوم", "مش قادر", "مش قادرة", "دايما", "بفضل أفكر")
    sleep_terms = ("مش عارف أنام", "مش عارفة أنام", "مش بعرف أنام", "أرق", "النوم", "انام")
    emotion_terms = ("مخنوق", "مخنوقة", "قلقان", "قلقانة", "زعلان", "زعلانة", "تعبان نفسيا", "خايف")
    knowledge_terms = ("ما هو", "ما هي", "اشرح", "يعني إيه", "يعني ايه", "كيف", "ليه")
    vague_terms = ("؟", "??", "هكذا", "مش عارف", "مش عارفة", "مش عارفه")

    def classify(self, message: str) -> IntentResult:
        text = message.strip().lower()
        if self._needs_clarification(text):
            return IntentResult("clarification_needed", 0.88, "Message is too vague to answer safely")
        if any(term in text for term in self.crisis_terms):
            return IntentResult("crisis_or_safety", 0.98, "Safety risk term detected")
        if any(term in text for term in self.rag_terms):
            return IntentResult("needs_rag", 0.9, "Reference material requested")
        if any(term in text for term in self.decision_terms):
            return IntentResult("decision_problem", 0.78, "Decision wording detected")
        if any(term in text for term in self.deep_terms) and (any(term in text for term in self.sleep_terms) or any(term in text for term in self.emotion_terms)):
            return IntentResult("deep_problem", 0.82, "Repeated or extended problem detected")
        if any(term in text for term in self.sleep_terms):
            return IntentResult("simple_advice", 0.84, "Simple sleep advice request")
        if any(term in text for term in self.emotion_terms):
            return IntentResult("emotional_support", 0.72, "Emotional state detected")
        if any(term in text for term in self.knowledge_terms) or text.endswith("?"):
            return IntentResult("knowledge_question", 0.7, "General knowledge question")
        return IntentResult("casual_chat", 0.68, "Default conversational message")

    def _needs_clarification(self, text: str) -> bool:
        if not text:
            return True
        if text in {"؟", "?", "??", "؟؟"}:
            return True
        if text in self.vague_terms:
            return True
        if text in {"مش عارف", "مش عارفة", "مش عارفه"}:
            return True
        words = text.split()
        if len(words) <= 2 and not any(
            term in text
            for term in (
                *self.crisis_terms,
                *self.rag_terms,
                *self.decision_terms,
                *self.sleep_terms,
                *self.emotion_terms,
                *self.knowledge_terms,
                "زهقان",
                "زهقانة",
            )
        ):
            return True
        return False


class IntensityService:
    def classify(self, intent: Intent, message: str) -> int:
        text = message.lower()
        if intent == "crisis_or_safety":
            return 3
        if intent == "deep_problem":
            return 4
        if intent == "decision_problem":
            return 4
        if intent == "emotional_support":
            return 3
        if intent == "simple_advice":
            return 2
        if intent == "clarification_needed":
            return 1
        if "بتهرب" in text or "مش عايز أواجه" in text:
            return 4
        return 1

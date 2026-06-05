from dataclasses import dataclass

from app.schemas.chat import Intent


@dataclass(frozen=True)
class FastIntentHints:
    likely_casual: bool
    likely_simple_advice: bool
    likely_unclear: bool
    likely_decision: bool
    likely_rag_request: bool
    likely_deep_problem: bool

    @property
    def intent_hint(self) -> Intent:
        if self.likely_rag_request:
            return "needs_rag"
        if self.likely_unclear:
            return "clarification_needed"
        if self.likely_decision:
            return "decision_problem"
        if self.likely_deep_problem:
            return "deep_problem"
        if self.likely_simple_advice:
            return "simple_advice"
        return "casual_chat"


class FastIntentHintService:
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
    sleep_terms = ("مش عارف أنام", "مش عارفة أنام", "مش بعرف أنام", "أرق", "النوم", "انام")
    decision_terms = ("اختار", "أقرر", "قرار", "أعمل إيه", "اعمل ايه", "محتار", "محتارة", "أسيب", "اسيب", "أكمل", "اكمل")
    deep_terms = ("بقالي", "شهر", "أسابيع", "اسابيع", "كل يوم", "مش قادر", "مش قادرة", "دايما", "بفضل أفكر")
    emotion_terms = ("مخنوق", "مخنوقة", "قلقان", "قلقانة", "زعلان", "زعلانة", "تعبان نفسيا", "خايف")
    casual_terms = ("زهقان", "زهقانة", "عامل ايه", "عاملة ايه", "هاي", "مرحبا", "ازيك")
    vague_terms = {"؟", "?", "??", "؟؟", "هكذا", "مش عارف", "مش عارفة", "مش عارفه"}

    def analyze(self, message: str) -> FastIntentHints:
        text = message.strip().lower()
        likely_rag_request = any(term in text for term in self.rag_terms)
        likely_simple_advice = any(term in text for term in self.sleep_terms)
        likely_decision = any(term in text for term in self.decision_terms)
        likely_deep_problem = any(term in text for term in self.deep_terms) and (
            likely_simple_advice or any(term in text for term in self.emotion_terms)
        )
        words = text.split()
        known = (
            likely_rag_request
            or likely_simple_advice
            or likely_decision
            or likely_deep_problem
            or any(term in text for term in self.emotion_terms)
            or any(term in text for term in self.casual_terms)
        )
        likely_unclear = not text or text in self.vague_terms or (len(words) <= 2 and not known)
        likely_casual = not any((likely_simple_advice, likely_unclear, likely_decision, likely_rag_request, likely_deep_problem))
        return FastIntentHints(
            likely_casual=likely_casual,
            likely_simple_advice=likely_simple_advice,
            likely_unclear=likely_unclear,
            likely_decision=likely_decision,
            likely_rag_request=likely_rag_request,
            likely_deep_problem=likely_deep_problem,
        )

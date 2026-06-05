import re


class QualityService:
    generic_patterns = ("أنا أفهم", "أفهم شعورك", "كل شيء سيكون بخير", "لا تقلق")
    overly_soft_patterns = ("حبيبتي", "يا روحي", "كل حاجة هتبقى تمام", "متزعليش")
    aggressive_patterns = ("غبية", "كسولة", "عيب عليك", "مشكلتك إنك")
    masculine_address_patterns = ("أنتَ", "عليكَ", "لكَ", "معكَ", "قُل لي")
    arabic_pattern = re.compile(r"[\u0600-\u06FF]")

    def check(
        self,
        *,
        reply: str,
        intent: str,
        used_rag: bool,
        rag_required: bool,
        rag_context: list[dict] | None = None,
    ) -> tuple[bool, list[str]]:
        issues: list[str] = []
        rag_context = rag_context or []
        arabic_chars = len(self.arabic_pattern.findall(reply))
        visible_chars = len([char for char in reply if not char.isspace()])
        if visible_chars and arabic_chars / visible_chars < 0.55:
            issues.append("not_arabic_enough")
        if any(pattern in reply for pattern in self.masculine_address_patterns):
            issues.append("not_feminine_voice")
        if any(pattern in reply for pattern in self.generic_patterns):
            issues.append("generic_support")
        if any(pattern in reply for pattern in self.overly_soft_patterns):
            issues.append("overly_soft")
        if any(pattern in reply for pattern in self.aggressive_patterns):
            issues.append("too_aggressive")
        if reply.count("?") + reply.count("؟") > 1:
            issues.append("too_many_questions")
        if intent == "casual_chat" and len(reply.split()) > 80:
            issues.append("casual_overanalyzed")
        if used_rag and not rag_required:
            issues.append("unnecessary_rag")
        if used_rag and rag_required and not rag_context and not any(term in reply for term in ("مش لاقية", "غير كاف", "غير كافي", "ارفع")):
            issues.append("unsupported_rag_claim")
        return not issues, issues

    def revise_once(self, reply: str, issues: list[str]) -> str:
        revised = reply
        if "too_many_questions" in issues:
            parts = re.split(r"([؟?])", revised)
            seen = 0
            kept: list[str] = []
            for item in parts:
                if item in {"؟", "?"}:
                    seen += 1
                    if seen > 1:
                        continue
                kept.append(item)
            revised = "".join(kept)
        if "generic_support" in issues:
            for pattern in self.generic_patterns:
                revised = revised.replace(pattern, "").strip()
        if "overly_soft" in issues:
            for pattern in self.overly_soft_patterns:
                revised = revised.replace(pattern, "").strip()
        if "too_aggressive" in issues:
            for pattern in self.aggressive_patterns:
                revised = revised.replace(pattern, "").strip()
        if "casual_overanalyzed" in issues:
            revised = " ".join(revised.split()[:35])
        if "unsupported_rag_claim" in issues:
            revised = ""
        return revised.strip()

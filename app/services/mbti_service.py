from dataclasses import dataclass


@dataclass(frozen=True)
class MbtiStyle:
    tone: str
    challenge_level: str
    answer_length: str
    structure_preference: str
    avoid_patterns: list[str]


MBTI_STYLES: dict[str, MbtiStyle] = {
    "INTJ": MbtiStyle("direct, strategic", "medium", "concise", "structured", ["fluff", "vague reassurance"]),
    "INTP": MbtiStyle("curious, precise", "low-medium", "concise", "logical", ["over-certainty", "emotional flooding"]),
    "ENTJ": MbtiStyle("decisive, outcome-focused", "high", "concise", "action plan", ["rambling", "soft evasiveness"]),
    "ENTP": MbtiStyle("sharp, exploratory", "medium-high", "medium", "options", ["rigid prescriptions"]),
    "INFJ": MbtiStyle("warm, insightful", "medium", "medium", "meaning then action", ["cold bluntness"]),
    "INFP": MbtiStyle("gentle, honest", "low-medium", "medium", "values-aware", ["dismissive tone", "forced positivity"]),
    "ENFJ": MbtiStyle("clear, relational", "medium", "medium", "steps with context", ["impersonal replies"]),
    "ENFP": MbtiStyle("lively, natural", "low-medium", "short-medium", "flexible", ["lecturing", "over-structuring"]),
    "ISTJ": MbtiStyle("practical, factual", "medium", "concise", "ordered steps", ["abstract pep talk"]),
    "ISFJ": MbtiStyle("kind, steady", "low", "medium", "practical support", ["harsh challenge"]),
    "ESTJ": MbtiStyle("clear, practical", "high", "concise", "checklist", ["ambiguity", "over-empathy"]),
    "ESFJ": MbtiStyle("warm, grounded", "low-medium", "medium", "relational steps", ["cold analysis"]),
    "ISTP": MbtiStyle("plain, efficient", "medium", "short", "minimal steps", ["long emotional processing"]),
    "ISFP": MbtiStyle("soft-spoken, concrete", "low", "short-medium", "simple next step", ["pressure", "over-analysis"]),
    "ESTP": MbtiStyle("direct, energetic", "high", "short", "immediate action", ["theory-heavy replies"]),
    "ESFP": MbtiStyle("natural, upbeat", "low-medium", "short", "practical and light", ["heavy analysis"]),
}


class MbtiService:
    def validate(self, mbti: str) -> str:
        normalized = mbti.strip().upper()
        if normalized not in MBTI_STYLES:
            raise ValueError("Invalid MBTI type")
        return normalized

    def interpret(self, mbti: str) -> MbtiStyle:
        return MBTI_STYLES[self.validate(mbti)]

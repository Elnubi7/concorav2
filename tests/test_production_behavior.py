import pytest
from pydantic import ValidationError

from app.core.errors import AppError
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.quality_service import QualityService


def chat(db, settings, message: str, mbti: str):
    return ChatService(settings, db).chat(ChatRequest(user_id="behavior-user", message=message, mbti=mbti, metadata={}))


def test_simple_chat_returns_model_generated_content(db, settings, monkeypatch):
    def fake_simple_reply(self, **kwargs):
        return f"generated: {kwargs['message']}"

    monkeypatch.setattr("app.services.llm_service.LLMService.generate_simple_reply_sync", fake_simple_reply)
    response = chat(db, settings, "زهقانة", "ENFP")

    assert response.path_used == "simple"
    assert response.reply == "generated: زهقانة"
    assert response.used_rag is False


def test_file_request_when_rag_disabled_stays_simple(db, settings, monkeypatch):
    calls = []

    def fake_simple_reply(self, **kwargs):
        calls.append(kwargs)
        return "generated disabled retrieval message"

    monkeypatch.setattr(settings, "enable_rag", False)
    monkeypatch.setattr("app.services.llm_service.LLMService.generate_simple_reply_sync", fake_simple_reply)
    response = chat(db, settings, "حسب الملف المرفوع، إيه الخلاصة؟", "INTJ")

    assert response.intent == "needs_rag"
    assert response.path_used == "simple"
    assert response.used_rag is False
    assert calls[0]["rag_disabled"] is True


def test_invalid_mbti_fails_without_server_crash(db, settings):
    with pytest.raises((AppError, ValidationError)):
        chat(db, settings, "أنا مش عارف أنام", "ABC")


def test_missing_mbti_matches_api_contract_without_server_crash():
    with pytest.raises(ValidationError):
        ChatRequest(user_id="behavior-user", message="أنا مش عارف أنام", metadata={})


def test_quality_guard_catches_required_failure_modes():
    service = QualityService()

    cases = [
        (
            {"reply": "This is an English answer.", "intent": "casual_chat", "used_rag": False, "rag_required": False},
            "not_arabic_enough",
        ),
        (
            {"reply": "أنا أفهم شعورك وكل شيء سيكون بخير", "intent": "emotional_support", "used_rag": False, "rag_required": False},
            "generic_support",
        ),
        (
            {"reply": "تعملي إيه؟ وليه؟", "intent": "simple_advice", "used_rag": False, "rag_required": False},
            "too_many_questions",
        ),
        (
            {"reply": "رجعت للملف.", "intent": "casual_chat", "used_rag": True, "rag_required": False},
            "unnecessary_rag",
        ),
        (
            {"reply": "أنتَ محتاج تهدى وتكتب الفكرة.", "intent": "simple_advice", "used_rag": False, "rag_required": False},
            "not_feminine_voice",
        ),
    ]

    for kwargs, expected_issue in cases:
        ok, issues = service.check(**kwargs)
        assert ok is False
        assert expected_issue in issues

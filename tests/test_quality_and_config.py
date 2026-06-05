from pathlib import Path

from app.core.config import get_settings
from app.services.quality_service import QualityService


def test_quality_guard_revises_extra_questions():
    service = QualityService()
    ok, issues = service.check(reply="تعملي خطوة صغيرة؟ وبعدها نشوف؟", intent="simple_advice", used_rag=False, rag_required=False)
    assert ok is False
    assert "too_many_questions" in issues
    revised = service.revise_once("تعملي خطوة صغيرة؟ وبعدها نشوف؟", issues)
    assert revised.count("؟") <= 1


def test_environment_config_loading():
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.app_name == "gemma-test"
    assert settings.llm_model == "test-model"
    assert settings.cors_origins == ["http://testserver"]


def test_no_hardcoded_secret_markers_in_code():
    root = Path(__file__).resolve().parents[1]
    forbidden = ("sk" + "-", "postgresql://" + "user:password", "hardcoded" + "-api-key")
    for path in root.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in forbidden), path

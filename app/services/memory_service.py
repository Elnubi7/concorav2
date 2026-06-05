from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.repositories.memory_repository import MemoryRepository


class MemoryService:
    def __init__(self, settings: Settings, db: Session) -> None:
        self.settings = settings
        self.repo = MemoryRepository(db)

    def load(self, user_id: str) -> dict:
        if not self.settings.memory_enabled:
            return {}
        profile = self.repo.get_profile(user_id)
        memories = self.repo.list_memories(user_id)
        return {
            "profile": profile.stable_profile if profile else {},
            "preferences": profile.preferences if profile else {},
            "memories": [{"key": item.key, "value": item.value, "confidence": item.confidence} for item in memories],
        }

    def maybe_write(self, user_id: str, message: str) -> None:
        if not self.settings.memory_enabled:
            return
        extracted = self._extract_stable_preference(message)
        if extracted:
            self.repo.add_memory(user_id=user_id, key=extracted["key"], value={"text": extracted["text"]}, confidence=0.75)

    def _extract_stable_preference(self, message: str) -> dict | None:
        normalized = message.strip()
        lowered = normalized.lower()
        stable_markers = (
            "أفضل أن",
            "افضل ان",
            "أفضل الردود",
            "افضل الردود",
            "بحب الردود",
            "اسمي",
            "ناديني",
            "أنا شغلي",
            "انا شغلي",
        )
        if any(marker.lower() in lowered for marker in stable_markers):
            return {"key": "stable_preference", "text": normalized[:500]}
        return None

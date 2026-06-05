from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Memory, UserProfile


class MemoryRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_profile(self, user_id: str) -> UserProfile | None:
        return self.db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))

    def upsert_profile_preferences(self, user_id: str, preferences: dict) -> UserProfile:
        profile = self.get_profile(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id, preferences=preferences, stable_profile={})
            self.db.add(profile)
        else:
            profile.preferences = {**(profile.preferences or {}), **preferences}
        self.db.flush()
        return profile

    def list_memories(self, user_id: str) -> list[Memory]:
        return list(self.db.scalars(select(Memory).where(Memory.user_id == user_id)).all())

    def add_memory(self, user_id: str, key: str, value: dict, confidence: float = 0.8) -> Memory:
        memory = Memory(user_id=user_id, key=key, value=value, confidence=confidence)
        self.db.add(memory)
        self.db.flush()
        return memory

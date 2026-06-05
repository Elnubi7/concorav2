from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ChatSession, GraphRun, Message, User


class ChatRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def ensure_user(self, user_id: str, metadata: dict | None = None) -> User:
        user = self.db.get(User, user_id)
        if user:
            return user
        user = User(id=user_id, metadata_json=metadata or {})
        self.db.add(user)
        self.db.flush()
        return user

    def get_or_create_session(self, user_id: str, session_id: str | None, metadata: dict | None = None) -> ChatSession:
        self.ensure_user(user_id)
        if session_id:
            session = self.db.get(ChatSession, session_id)
            if session:
                return session
        kwargs = {"user_id": user_id, "metadata_json": metadata or {}}
        if session_id:
            kwargs["id"] = session_id
        session = ChatSession(**kwargs)
        self.db.add(session)
        self.db.flush()
        return session

    def add_message(
        self,
        *,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        intent: str | None = None,
        intensity: int | None = None,
        used_rag: bool = False,
        mbti: str | None = None,
        metadata: dict | None = None,
    ) -> Message:
        message = Message(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
            intensity=intensity,
            used_rag=used_rag,
            mbti=mbti,
            metadata_json=metadata or {},
        )
        self.db.add(message)
        self.db.flush()
        return message

    def list_session_messages(self, session_id: str, limit: int = 50) -> list[Message]:
        stmt = select(Message).where(Message.session_id == session_id).order_by(Message.created_at.asc()).limit(limit)
        return list(self.db.scalars(stmt).all())

    def get_session(self, session_id: str) -> ChatSession | None:
        return self.db.get(ChatSession, session_id)

    def archive_or_delete_session(self, session_id: str, archive: bool) -> bool:
        session = self.db.get(ChatSession, session_id)
        if not session:
            return False
        if archive:
            session.is_archived = True
        else:
            self.db.delete(session)
        self.db.flush()
        return True

    def add_graph_run(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str | None,
        final_intent: str | None,
        intensity: int | None,
        used_rag: bool,
        node_latencies_ms: dict,
        metadata: dict | None = None,
    ) -> GraphRun:
        run = GraphRun(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            final_intent=final_intent,
            intensity=intensity,
            used_rag=used_rag,
            node_latencies_ms=node_latencies_ms,
            metadata_json=metadata or {},
        )
        self.db.add(run)
        self.db.flush()
        return run

from uuid import uuid4

from app.db.repositories.chat_repository import ChatRepository
from app.graph.state import GraphState
from app.services.memory_service import MemoryService


def session_loader(state: GraphState) -> GraphState:
    if state.get("db") is None:
        state["session_id"] = state.get("session_id") or f"graph-{uuid4()}"
        state["user_message_id"] = f"graph-{uuid4()}"
        state["session_messages"] = []
        state["memory"] = {}
        return state
    repo = ChatRepository(state["db"])
    session = repo.get_or_create_session(state["user_id"], state.get("session_id"), state.get("metadata", {}))
    state["session_id"] = session.id
    user_message = repo.add_message(
        user_id=state["user_id"],
        session_id=session.id,
        role="user",
        content=state["message"],
        mbti=state.get("mbti"),
        metadata=state.get("metadata", {}),
    )
    state["user_message_id"] = user_message.id
    state["session_messages"] = [
        {"role": msg.role, "content": msg.content} for msg in repo.list_session_messages(session.id, limit=30)
    ]
    state["memory"] = MemoryService(state["settings"], state["db"]).load(state["user_id"])
    return state

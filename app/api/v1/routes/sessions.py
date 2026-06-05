from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.schemas.chat import SessionOut
from app.services.chat_service import ChatService

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}", response_model=SessionOut)
def get_session(session_id: str, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> SessionOut:
    session = ChatService(settings, db).get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> dict[str, bool]:
    deleted = ChatService(settings, db).delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}

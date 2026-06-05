from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import request_id_ctx
from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.services.metrics_service import MetricsService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request_body: ChatRequest, request: Request, db: Session | None = Depends(get_db), settings: Settings = Depends(get_settings)) -> ChatResponse:
    MetricsService.increment_request()
    return ChatService(settings, db).chat(request_body, request_id=getattr(request.state, "request_id", request_id_ctx.get()))


@router.post("/stream")
async def chat_stream(
    request_body: ChatRequest,
    request: Request,
    db: Session | None = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    if not settings.streaming_enabled:
        raise HTTPException(status_code=404, detail="Streaming is disabled")
    service = ChatService(settings, db)
    response = service.chat(request_body, request_id=getattr(request.state, "request_id", request_id_ctx.get()))
    return StreamingResponse(
        service.stream_response(response),
        media_type="text/plain; charset=utf-8",
    )

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.db.session import get_db
from app.schemas.chat import DocumentUploadRequest, DocumentUploadResponse, ReindexResponse
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingProviderError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse)
def upload_document(request: DocumentUploadRequest, db: Session | None = Depends(get_db), settings: Settings = Depends(get_settings)) -> DocumentUploadResponse:
    if not settings.enable_rag:
        raise AppError("Document retrieval is disabled", 404)
    if db is None:
        raise AppError("Database is required for document retrieval", 503)
    try:
        document_id, count = DocumentService(db, settings).upload(
            title=request.title,
            content=request.content,
            user_id=request.user_id,
            source=request.source,
            metadata=request.metadata,
        )
    except ValueError as exc:
        raise AppError(str(exc), 422) from exc
    except EmbeddingProviderError as exc:
        raise AppError("Embedding provider unavailable", 503) from exc
    db.commit()
    return DocumentUploadResponse(document_id=document_id, chunks_indexed=count)


@router.post("/reindex", response_model=ReindexResponse)
def reindex_documents(db: Session | None = Depends(get_db), settings: Settings = Depends(get_settings)) -> ReindexResponse:
    if not settings.enable_rag:
        raise AppError("Document retrieval is disabled", 404)
    if db is None:
        raise AppError("Database is required for document retrieval", 503)
    try:
        documents_seen, chunks_indexed = DocumentService(db, settings).reindex()
    except EmbeddingProviderError as exc:
        raise AppError("Embedding provider unavailable", 503) from exc
    db.commit()
    return ReindexResponse(documents_seen=documents_seen, chunks_indexed=chunks_indexed)

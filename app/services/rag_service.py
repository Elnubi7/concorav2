from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.repositories.document_repository import DocumentRepository
from app.services.embedding_service import EmbeddingProviderError, EmbeddingService


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict


class BaseRetriever:
    def retrieve(self, query: str, user_id: str | None, top_k: int, score_threshold: float) -> list[RetrievalResult]:
        raise NotImplementedError


class PgVectorRetriever(BaseRetriever):
    def __init__(self, settings: Settings, db: Session) -> None:
        self.settings = settings
        self.db = db

    def retrieve(self, query: str, user_id: str | None, top_k: int, score_threshold: float) -> list[RetrievalResult]:
        repo = DocumentRepository(self.db)
        query_embedding = EmbeddingService(self.settings).embed_query(query)
        rows = repo.semantic_search_pgvector(query_embedding, user_id, top_k, score_threshold)
        return [RetrievalResult(**row) for row in rows]


class MemoryRetriever(BaseRetriever):
    def __init__(self, settings: Settings, db: Session) -> None:
        self.settings = settings
        self.db = db

    def retrieve(self, query: str, user_id: str | None, top_k: int, score_threshold: float) -> list[RetrievalResult]:
        repo = DocumentRepository(self.db)
        query_embedding = EmbeddingService(self.settings).embed_query(query)
        rows = repo.semantic_search_memory(query_embedding, user_id, top_k, score_threshold)
        return [RetrievalResult(**row) for row in rows]


class UnsupportedRetriever(BaseRetriever):
    def retrieve(self, query: str, user_id: str | None, top_k: int, score_threshold: float) -> list[RetrievalResult]:
        _ = (query, user_id, top_k, score_threshold)
        raise NotImplementedError("Vector backend is not implemented")


class RagService:
    def __init__(self, settings: Settings, db: Session) -> None:
        self.settings = settings
        self.db = db
        self.retriever: BaseRetriever = self._build_retriever()

    def _build_retriever(self) -> BaseRetriever:
        if self.settings.vector_backend == "pgvector":
            return PgVectorRetriever(self.settings, self.db)
        if self.settings.vector_backend == "memory":
            return MemoryRetriever(self.settings, self.db)
        return UnsupportedRetriever()

    def should_retrieve(self, *, intent: str, message: str) -> bool:
        if not self.settings.enable_rag:
            return False
        text = message.lower()
        explicit = any(
            term in text
            for term in (
                "الملف",
                "المستند",
                "الوثيقة",
                "الداتا",
                "ارجعي لـ",
                "ارجعي ل",
                "بناءً على اللي رفعته",
                "بناء على اللي رفعته",
                "حسب المرجع",
                "knowledge base",
                "document",
                "uploaded file",
                "reference",
            )
        )
        return intent == "needs_rag" or explicit

    def retrieve(self, *, query: str, user_id: str, session_id: str) -> list[dict]:
        if not self.settings.enable_rag:
            return []
        try:
            results = self.retriever.retrieve(
                query=query,
                user_id=user_id,
                top_k=self.settings.rag_top_k,
                score_threshold=self.settings.rag_score_threshold,
            )
        except EmbeddingProviderError:
            raise
        payload = [result.__dict__ for result in results]
        DocumentRepository(self.db).log_rag_query(
            user_id=user_id,
            session_id=session_id,
            query=query,
            top_k=self.settings.rag_top_k,
            score_threshold=self.settings.rag_score_threshold,
            results=payload,
        )
        return payload

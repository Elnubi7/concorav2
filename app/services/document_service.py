from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.repositories.document_repository import DocumentRepository
from app.services.embedding_service import EmbeddingService


class DocumentService:
    def __init__(self, db: Session, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.repo = DocumentRepository(db)
        self.embedding_service = EmbeddingService(self.settings)

    def chunk_text(self, content: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
        chunk_size = chunk_size or self.settings.rag_chunk_size
        chunk_overlap = self.settings.rag_chunk_overlap if chunk_overlap is None else chunk_overlap
        if chunk_size <= 0:
            raise ValueError("RAG_CHUNK_SIZE must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be lower than RAG_CHUNK_SIZE")
        words = content.split()
        chunks: list[str] = []
        step = chunk_size - chunk_overlap
        for index in range(0, len(words), step):
            chunk = " ".join(words[index : index + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks or [content]

    def upload(self, *, title: str, content: str, user_id: str | None, source: str | None, metadata: dict) -> tuple[str, int]:
        if len(content.encode("utf-8")) > self.settings.rag_max_document_size:
            raise ValueError("Document exceeds RAG_MAX_DOCUMENT_SIZE")
        document = self.repo.create_document(title=title, content=content, user_id=user_id, source=source, metadata=metadata)
        chunks = self.chunk_text(content)
        embeddings = self.embedding_service.embed_texts(chunks)
        indexed = self.repo.replace_chunks(document.id, chunks, embeddings)
        return document.id, indexed

    def reindex(self) -> tuple[int, int]:
        documents = self.repo.list_documents()
        total_chunks = 0
        for document in documents:
            raw_content = (document.metadata_json or {}).get("raw_content", "")
            chunks = self.chunk_text(raw_content)
            embeddings = self.embedding_service.embed_texts(chunks)
            total_chunks += self.repo.replace_chunks(document.id, chunks, embeddings)
        return len(documents), total_chunks

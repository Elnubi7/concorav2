import hashlib
import math

from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk, RagQuery


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_document(self, *, title: str, content: str, user_id: str | None, source: str | None, metadata: dict) -> Document:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        document = Document(
            title=title,
            user_id=user_id,
            source=source,
            content_hash=content_hash,
            metadata_json={**metadata, "raw_content": content},
        )
        self.db.add(document)
        self.db.flush()
        return document

    def replace_chunks(self, document_id: str, chunks: list[str], embeddings: list[list[float]] | None = None) -> int:
        self.db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        for index, chunk in enumerate(chunks):
            self.db.add(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=index,
                    content=chunk,
                    embedding=embeddings[index] if embeddings else None,
                )
            )
        self.db.flush()
        return len(chunks)

    def list_chunks(self, user_id: str | None = None) -> list[DocumentChunk]:
        stmt = select(DocumentChunk).join(Document, Document.id == DocumentChunk.document_id)
        if user_id:
            stmt = stmt.where((Document.user_id == user_id) | (Document.user_id.is_(None)))
        return list(self.db.scalars(stmt).all())

    def semantic_search_memory(self, query_embedding: list[float], user_id: str | None, top_k: int, score_threshold: float) -> list[dict]:
        results: list[dict] = []
        for chunk in self.list_chunks(user_id=user_id):
            if not chunk.embedding:
                continue
            score = self._cosine_similarity(query_embedding, list(chunk.embedding))
            if score >= score_threshold:
                results.append(
                    {
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "score": score,
                        "metadata": chunk.metadata_json or {},
                    }
                )
        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]

    def semantic_search_pgvector(self, query_embedding: list[float], user_id: str | None, top_k: int, score_threshold: float) -> list[dict]:
        embedding_literal = "[" + ",".join(str(value) for value in query_embedding) + "]"
        sql = """
            SELECT
                dc.id AS chunk_id,
                dc.document_id AS document_id,
                dc.content AS content,
                dc.metadata_json AS metadata,
                1 - (dc.embedding <=> CAST(:embedding AS vector)) AS score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.embedding IS NOT NULL
              AND (:user_id IS NULL OR d.user_id = :user_id OR d.user_id IS NULL)
              AND 1 - (dc.embedding <=> CAST(:embedding AS vector)) >= :score_threshold
            ORDER BY dc.embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """
        rows = self.db.execute(
            text(sql),
            {
                "embedding": embedding_literal,
                "user_id": user_id,
                "score_threshold": score_threshold,
                "top_k": top_k,
            },
        ).mappings()
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "score": float(row["score"]),
                "metadata": row["metadata"] or {},
            }
            for row in rows
        ]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        denominator = left_norm * right_norm
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def list_documents(self) -> list[Document]:
        return list(self.db.scalars(select(Document)).all())

    def log_rag_query(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        top_k: int,
        score_threshold: float,
        results: list[dict],
    ) -> RagQuery:
        row = RagQuery(
            user_id=user_id,
            session_id=session_id,
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            results=results,
        )
        self.db.add(row)
        self.db.flush()
        return row

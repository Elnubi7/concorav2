"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-04
"""

import sqlalchemy as sa

from alembic import op
from app.core.config import get_settings

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    dialect_name = op.get_bind().dialect.name
    embedding_type = sa.JSON()
    if dialect_name == "postgresql":
        from pgvector.sqlalchemy import Vector

        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        embedding_type = Vector(get_settings().embedding_dimensions)
    op.create_table("users", sa.Column("id", sa.String(length=64), primary_key=True), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()), sa.Column("metadata_json", sa.JSON(), nullable=False))
    op.create_table("documents", sa.Column("id", sa.String(length=64), primary_key=True), sa.Column("user_id", sa.String(length=64), sa.ForeignKey("users.id"), nullable=True), sa.Column("title", sa.String(length=255), nullable=False), sa.Column("source", sa.String(length=255), nullable=True), sa.Column("content_hash", sa.String(length=128), nullable=True), sa.Column("metadata_json", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()))
    op.create_table("sessions", sa.Column("id", sa.String(length=64), primary_key=True), sa.Column("user_id", sa.String(length=64), sa.ForeignKey("users.id"), nullable=False), sa.Column("title", sa.String(length=255), nullable=True), sa.Column("is_archived", sa.Boolean(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()), sa.Column("metadata_json", sa.JSON(), nullable=False))
    op.create_table("user_profiles", sa.Column("id", sa.String(length=64), primary_key=True), sa.Column("user_id", sa.String(length=64), sa.ForeignKey("users.id"), nullable=False, unique=True), sa.Column("stable_profile", sa.JSON(), nullable=False), sa.Column("preferences", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()))
    op.create_table("memories", sa.Column("id", sa.String(length=64), primary_key=True), sa.Column("user_id", sa.String(length=64), sa.ForeignKey("users.id"), nullable=False), sa.Column("key", sa.String(length=128), nullable=False), sa.Column("value", sa.JSON(), nullable=False), sa.Column("confidence", sa.Float(), nullable=False), sa.Column("source", sa.String(length=64), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()))
    op.create_table("messages", sa.Column("id", sa.String(length=64), primary_key=True), sa.Column("session_id", sa.String(length=64), sa.ForeignKey("sessions.id"), nullable=False), sa.Column("user_id", sa.String(length=64), sa.ForeignKey("users.id"), nullable=False), sa.Column("role", sa.String(length=32), nullable=False), sa.Column("content", sa.Text(), nullable=False), sa.Column("intent", sa.String(length=64), nullable=True), sa.Column("intensity", sa.Integer(), nullable=True), sa.Column("used_rag", sa.Boolean(), nullable=False), sa.Column("mbti", sa.String(length=8), nullable=True), sa.Column("metadata_json", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()))
    op.create_table("document_chunks", sa.Column("id", sa.String(length=64), primary_key=True), sa.Column("document_id", sa.String(length=64), sa.ForeignKey("documents.id"), nullable=False), sa.Column("chunk_index", sa.Integer(), nullable=False), sa.Column("content", sa.Text(), nullable=False), sa.Column("embedding", embedding_type, nullable=True), sa.Column("metadata_json", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()))
    op.create_table("rag_queries", sa.Column("id", sa.String(length=64), primary_key=True), sa.Column("user_id", sa.String(length=64), sa.ForeignKey("users.id"), nullable=False), sa.Column("session_id", sa.String(length=64), sa.ForeignKey("sessions.id"), nullable=False), sa.Column("query", sa.Text(), nullable=False), sa.Column("top_k", sa.Integer(), nullable=False), sa.Column("score_threshold", sa.Float(), nullable=False), sa.Column("results", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()))
    op.create_table("graph_runs", sa.Column("id", sa.String(length=64), primary_key=True), sa.Column("user_id", sa.String(length=64), sa.ForeignKey("users.id"), nullable=False), sa.Column("session_id", sa.String(length=64), sa.ForeignKey("sessions.id"), nullable=False), sa.Column("request_id", sa.String(length=128), nullable=True), sa.Column("final_intent", sa.String(length=64), nullable=True), sa.Column("intensity", sa.Integer(), nullable=True), sa.Column("used_rag", sa.Boolean(), nullable=False), sa.Column("node_latencies_ms", sa.JSON(), nullable=False), sa.Column("metadata_json", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()))
    op.create_table("evaluation_cases", sa.Column("id", sa.String(length=64), primary_key=True), sa.Column("name", sa.String(length=255), nullable=False), sa.Column("input_payload", sa.JSON(), nullable=False), sa.Column("expected", sa.JSON(), nullable=False), sa.Column("result", sa.JSON(), nullable=True), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()))
    op.create_index("ix_documents_user_id", "documents", ["user_id"])
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    op.create_index("ix_user_profiles_user_id", "user_profiles", ["user_id"], unique=True)
    op.create_index("ix_memories_user_id", "memories", ["user_id"])
    op.create_index("ix_memories_key", "memories", ["key"])
    op.create_index("ix_messages_session_id", "messages", ["session_id"])
    op.create_index("ix_messages_user_id", "messages", ["user_id"])
    op.create_index("ix_messages_role", "messages", ["role"])
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_document_chunks_doc_idx", "document_chunks", ["document_id", "chunk_index"])
    if dialect_name == "postgresql":
        op.create_index("ix_document_chunks_embedding_hnsw", "document_chunks", ["embedding"], postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"})
    op.create_index("ix_rag_queries_user_id", "rag_queries", ["user_id"])
    op.create_index("ix_rag_queries_session_id", "rag_queries", ["session_id"])
    op.create_index("ix_graph_runs_user_id", "graph_runs", ["user_id"])
    op.create_index("ix_graph_runs_session_id", "graph_runs", ["session_id"])


def downgrade() -> None:
    for table in ("evaluation_cases", "graph_runs", "rag_queries", "document_chunks", "messages", "memories", "user_profiles", "sessions", "documents", "users"):
        op.drop_table(table)

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def create_db_engine() -> Engine:
    settings = get_settings()
    if not settings.enable_db:
        raise RuntimeError("Database is disabled")
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required when ENABLE_DB=true")
    if settings.database_url.startswith("sqlite"):
        return create_engine(settings.database_url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    return create_engine(settings.database_url, pool_pre_ping=True)


@lru_cache
def get_engine() -> Engine:
    return create_db_engine()


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autocommit=False, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session | None, None, None]:
    if not get_settings().enable_db:
        yield None
        return
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()


def reset_db_engine_cache() -> None:
    get_sessionmaker.cache_clear()
    get_engine.cache_clear()

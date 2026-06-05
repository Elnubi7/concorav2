from fastapi import APIRouter, Depends

try:
    from redis import Redis
except ModuleNotFoundError:  # pragma: no cover
    Redis = None
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.security import protect_metrics
from app.db.session import get_db
from app.schemas.chat import HealthResponse, MetricsResponse, ReadyResponse
from app.services.metrics_service import MetricsService

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", app_name=settings.app_name, app_version=settings.app_version)


@router.get("/ready", response_model=ReadyResponse)
def ready(db: Session | None = Depends(get_db), settings: Settings = Depends(get_settings)) -> ReadyResponse:
    database_required = settings.enable_db or settings.enable_rag or settings.enable_memory
    redis_required = settings.enable_redis or settings.enable_memory
    database_status = "skipped"
    redis_status = "skipped"
    if database_required:
        database_status = "ok"
        try:
            if db is None:
                raise RuntimeError("database session unavailable")
            db.execute(text("SELECT 1"))
        except Exception:
            database_status = "error"
    if redis_required:
        redis_status = "ok"
        try:
            if Redis is None:
                raise RuntimeError("redis package not installed")
            Redis.from_url(settings.redis_url, socket_connect_timeout=0.25).ping()
        except Exception:
            redis_status = "error"
    status = "ok" if database_status == "ok" and redis_status == "ok" else "degraded"
    if database_status == "skipped" and redis_status == "skipped":
        status = "ok"
    elif database_status in {"ok", "skipped"} and redis_status in {"ok", "skipped"}:
        status = "ok"
    return ReadyResponse(status=status, database=database_status, redis=redis_status)


@router.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(protect_metrics)])
def metrics() -> MetricsResponse:
    return MetricsResponse(
        requests_total=MetricsService.requests_total,
        graph_runs_total=MetricsService.graph_runs_total,
        rag_queries_total=MetricsService.rag_queries_total,
    )

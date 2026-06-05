from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import RuntimeConfigError, get_settings, validate_runtime_settings
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging, get_logger, request_id_ctx
from app.core.rate_limit import RateLimiter

logger = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rate_limiter = RateLimiter(settings)

    @app.on_event("startup")
    async def validate_runtime_configuration() -> None:
        try:
            validate_runtime_settings(settings)
        except RuntimeConfigError as exc:
            logger.error("runtime_configuration_invalid errors=%s", exc.errors)
            raise RuntimeError("Invalid runtime configuration: " + "; ".join(exc.errors)) from exc

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid4()))
        request.state.request_id = request_id
        token = request_id_ctx.set(request_id)
        try:
            await rate_limiter.check(request)
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response
        finally:
            request_id_ctx.reset(token)

    register_error_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()

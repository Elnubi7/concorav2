import logging
import sys
from contextvars import ContextVar

try:
    import structlog
except ModuleNotFoundError:  # pragma: no cover - dependency fallback for minimal local test environments
    structlog = None

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


def add_request_id(_: object, __: str, event_dict: dict) -> dict:
    request_id = request_id_ctx.get()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def configure_logging(level: str) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level.upper(), logging.INFO))
    if structlog is None:
        return
    structlog.configure(
        processors=[
            add_request_id,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    if structlog is None:
        return logging.getLogger(name)
    return structlog.get_logger(name)

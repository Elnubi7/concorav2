from fastapi import Header, HTTPException, status

from app.core.config import get_settings


async def auth_placeholder(authorization: str | None = Header(default=None)) -> None:
    _ = authorization
    return None


async def protect_metrics(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.metrics_token:
        return None
    expected = f"Bearer {settings.metrics_token}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return None

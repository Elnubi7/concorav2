from fastapi import Request

from app.core.config import Settings


class RateLimiter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def check(self, request: Request) -> None:
        _ = request
        return None

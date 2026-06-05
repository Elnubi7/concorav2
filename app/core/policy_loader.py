import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache
def load_policy(name: str) -> dict[str, Any]:
    path = Path("app/config") / f"{name}.yml"
    return json.loads(path.read_text(encoding="utf-8"))

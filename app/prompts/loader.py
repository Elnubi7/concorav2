from functools import lru_cache
from pathlib import Path


@lru_cache
def load_prompt(name: str, version: str = "v1") -> str:
    root = Path(__file__).resolve().parent
    path = root / version / f"{name}.md"
    return path.read_text(encoding="utf-8")

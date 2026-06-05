import time
from collections.abc import Callable
from typing import Any

from app.graph.state import GraphState


def timed_node(name: str, func: Callable[[GraphState], GraphState]) -> Callable[[GraphState], GraphState]:
    def wrapper(state: GraphState) -> GraphState:
        start = time.perf_counter()
        updated = func(state)
        latencies = dict(updated.get("node_latencies_ms", {}))
        latencies[name] = round((time.perf_counter() - start) * 1000, 3)
        updated["node_latencies_ms"] = latencies
        return updated

    return wrapper


def get_service(state: GraphState, key: str) -> Any:
    return state["metadata"]["services"][key]

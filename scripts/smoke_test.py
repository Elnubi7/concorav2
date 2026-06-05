import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


def request_json(method: str, url: str, payload: dict | None = None, timeout: float = 60) -> tuple[int, dict | str, float]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            try:
                parsed: dict | str = json.loads(body) if body else {}
            except json.JSONDecodeError:
                parsed = body
            return response.status, parsed, elapsed_ms(start)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed: dict | str = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return exc.code, parsed, elapsed_ms(start)
    except urllib.error.URLError as exc:
        return 0, str(exc), elapsed_ms(start)


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def preview(body: object) -> str:
    if not isinstance(body, dict):
        return str(body)[:160]
    reply = str(body.get("reply", ""))
    return reply[:120]


def chat_detail(body: object) -> str:
    if not isinstance(body, dict):
        return str(body)
    return (
        f"path={body.get('path_used')} reason={body.get('route_reason')} "
        f"intent={body.get('intent')} used_rag={body.get('used_rag')} preview={preview(body)}"
    )


def check(name: str, condition: bool, elapsed: float, detail: object) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"{status} {name}: {elapsed}ms {detail}")
    return condition


def check_chat(name: str, condition: bool, elapsed: float, body: object) -> bool:
    ok = check(name, condition, elapsed, chat_detail(body))
    if not ok:
        print(f"{name} response body: {body}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma API smoke test")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    timeout = float(os.environ.get("SMOKE_TIMEOUT_SECONDS", "60"))
    host = os.environ.get("API_HOST")
    port = os.environ.get("API_PORT")
    base_url = args.base_url or os.environ.get("BASE_URL") or os.environ.get("SMOKE_BASE_URL") or f"http://{host}:{port}"
    if not host or not port:
        print("FAIL config: API_HOST and API_PORT must be set")
        return 1

    checks: list[bool] = []

    status, body, elapsed = request_json("GET", f"{base_url}/api/v1/health", timeout=timeout)
    checks.append(check("health", status == 200 and isinstance(body, dict) and body.get("status") == "ok", elapsed, body))

    status, body, elapsed = request_json("GET", f"{base_url}/api/v1/ready", timeout=timeout)
    checks.append(check("ready", status == 200 and isinstance(body, dict) and body.get("status") in {"ok", "degraded"}, elapsed, body))

    status, body, elapsed = request_json(
        "POST",
        f"{base_url}/api/v1/chat",
        {"user_id": "smoke-user", "message": "زهقانة", "mbti": "ENFP", "metadata": {}},
        timeout=timeout,
    )
    checks.append(
        check_chat(
            "casual chat",
            status == 200
            and isinstance(body, dict)
            and body.get("used_rag") is False
            and body.get("path_used") == "simple",
            elapsed,
            body,
        )
    )

    status, body, elapsed = request_json(
        "POST",
        f"{base_url}/api/v1/chat",
        {
            "user_id": "smoke-user",
            "message": "حللي المشكلة خطوة خطوة: عندي شغل كتير ووقت قليل ومش عارفة أبدأ منين",
            "mbti": "INTJ",
            "metadata": {},
        },
        timeout=timeout,
    )
    checks.append(
        check_chat(
            "complex chat",
            status == 200 and isinstance(body, dict) and body.get("path_used") in {"langgraph", "simple"},
            elapsed,
            body,
        )
    )

    if os.environ.get("ENABLE_RAG", "").lower() == "true":
        status, body, elapsed = request_json(
            "POST",
            f"{base_url}/api/v1/chat",
            {"user_id": "smoke-user", "message": "ارجعي للملف اللي رفعته وقولي الخلاصة", "mbti": "INTJ", "metadata": {}},
            timeout=timeout,
        )
        checks.append(
            check_chat(
                "rag chat",
                status == 200 and isinstance(body, dict) and body.get("path_used") == "rag",
                elapsed,
                body,
            )
        )

    if os.environ.get("STREAMING_ENABLED", "").lower() == "true":
        status, body, elapsed = request_json(
            "POST",
            f"{base_url}/api/v1/chat/stream",
            {"user_id": "smoke-user", "message": "زهقانة", "mbti": "ENFP", "metadata": {}},
            timeout=timeout,
        )
        checks.append(check("stream chat", status == 200, elapsed, preview(body)))

    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())

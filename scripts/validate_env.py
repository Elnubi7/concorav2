import argparse
import os
import sys
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings, runtime_settings_errors  # noqa: E402


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


def apply_profile(profile: str) -> None:
    if profile == "server":
        os.environ.setdefault("APP_ENV", "production")
    elif profile == "local":
        os.environ.setdefault("APP_ENV", "local")


def validate(profile: str | None = None) -> list[str]:
    if profile:
        apply_profile(profile)
    get_settings.cache_clear()
    try:
        settings = get_settings()
    except ValidationError as exc:
        return [f"Invalid environment value: {error['loc'][0]} - {error['msg']}" for error in exc.errors()]
    return runtime_settings_errors(settings)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Gemma runtime environment")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--profile", choices=["local", "server"], default=None)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    errors = validate(args.profile)
    if errors:
        print("FAIL environment validation")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS environment validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())

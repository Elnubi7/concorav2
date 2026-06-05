from dataclasses import dataclass

from app.core.config import Settings
from app.core.policy_loader import load_policy
from app.services.llm_service import LLMProviderError, LLMService


@dataclass(frozen=True)
class ResourceIntentResult:
    user_requested_resource: bool
    requested_media_types: list[str]
    requested_topic: str | None
    confidence: float
    reason: str


class ResourceIntentClassifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.policy = load_policy("resource_rag_policy")

    def classify(self, message: str) -> ResourceIntentResult:
        if self.settings.resource_intent_classifier == "deterministic":
            return self._deterministic(message)
        return self._semantic(message)

    def _semantic(self, message: str) -> ResourceIntentResult:
        model = self.settings.llm_model
        if self.settings.resource_intent_classifier_model != "use_default_llm":
            model = self.settings.resource_intent_classifier_model
        llm = LLMService(
            self.settings.llm_provider,
            model,
            self.settings.effective_llm_api_key,
            base_url=self.settings.effective_llm_base_url,
            timeout_seconds=self.settings.llm_timeout_seconds,
            max_retries=self.settings.llm_max_retries,
            app_env=self.settings.app_env,
        )
        try:
            payload = llm.classify_resource_intent_sync(
                message=message,
                max_tokens=min(self.settings.llm_max_tokens, 120),
                temperature=0,
            )
        except LLMProviderError:
            raise
        return ResourceIntentResult(
            user_requested_resource=bool(payload.get("user_requested_resource", False)),
            requested_media_types=[str(item) for item in payload.get("requested_media_types", [])],
            requested_topic=str(payload["requested_topic"]) if payload.get("requested_topic") else None,
            confidence=float(payload.get("confidence", 0.0)),
            reason=str(payload.get("reason", "")),
        )

    def _deterministic(self, message: str) -> ResourceIntentResult:
        text = message.lower()
        patterns = self.policy["deterministic_patterns"]
        requested = any(term.lower() in text for term in patterns["resource_request"])
        media_types = [
            media_type
            for media_type, terms in patterns["media_types"].items()
            if any(term.lower() in text for term in terms)
        ]
        return ResourceIntentResult(
            user_requested_resource=requested,
            requested_media_types=media_types,
            requested_topic=message if requested else None,
            confidence=1.0 if requested else 0.0,
            reason="deterministic_policy_match" if requested else "deterministic_no_match",
        )

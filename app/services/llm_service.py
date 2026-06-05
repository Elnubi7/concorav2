from collections.abc import AsyncIterator
from typing import Any

from app.core.logging import get_logger
from app.prompts.loader import load_prompt
from app.services.mbti_service import MbtiStyle

TEST_PROVIDER = "test"
OPENAI_PROVIDER = "openai"
OPENAI_COMPATIBLE_PROVIDER = "openai_compatible"
GITHUB_MODELS_PROVIDER = "github_models"

logger = get_logger(__name__)


class LLMProviderError(RuntimeError):
    pass


class LLMService:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        app_env: str = "production",
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.app_env = app_env

    def generate_gemma_reply_sync(
        self,
        *,
        message: str,
        intent: str,
        intensity: int,
        style: MbtiStyle,
        rag_context: list[dict],
        plan: dict,
    ) -> str:
        if self.provider == TEST_PROVIDER:
            if self.app_env != "test":
                raise LLMProviderError("Test LLM provider is only allowed when APP_ENV=test")
            return self._generate_test_reply(message=message, intent=intent, rag_context=rag_context)
        if self.provider in {OPENAI_PROVIDER, OPENAI_COMPATIBLE_PROVIDER, GITHUB_MODELS_PROVIDER}:
            return self._generate_openai_compatible_reply(
                message=message,
                intent=intent,
                intensity=intensity,
                style=style,
                rag_context=rag_context,
                plan=plan,
            )
        raise LLMProviderError("Unsupported LLM provider")

    def generate_fast_reply_sync(
        self,
        *,
        message: str,
        intent_hint: str,
        intensity_hint: int,
        clarification_may_be_needed: bool,
        style: MbtiStyle,
        session_summary: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if self.provider == TEST_PROVIDER:
            if self.app_env != "test":
                raise LLMProviderError("Test LLM provider is only allowed when APP_ENV=test")
            return self._generate_fast_test_reply(
                message=message,
                intent_hint=intent_hint,
                clarification_may_be_needed=clarification_may_be_needed,
                style=style,
            )
        if self.provider in {OPENAI_PROVIDER, OPENAI_COMPATIBLE_PROVIDER, GITHUB_MODELS_PROVIDER}:
            return self._generate_openai_compatible_fast_reply(
                message=message,
                intent_hint=intent_hint,
                intensity_hint=intensity_hint,
                clarification_may_be_needed=clarification_may_be_needed,
                style=style,
                session_summary=session_summary,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        raise LLMProviderError("Unsupported LLM provider")

    def generate_simple_reply_sync(
        self,
        *,
        message: str,
        rag_disabled: bool,
        intelligence: dict | None = None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if self.provider == TEST_PROVIDER:
            if self.app_env != "test":
                raise LLMProviderError("Test LLM provider is only allowed when APP_ENV=test")
            suffix = " rag-disabled" if rag_disabled else ""
            return f"test provider reply{suffix}: {message}"
        if self.provider in {OPENAI_COMPATIBLE_PROVIDER, GITHUB_MODELS_PROVIDER}:
            return self._generate_openai_compatible_simple_reply(
                message=message,
                rag_disabled=rag_disabled,
                intelligence=intelligence or {},
                max_tokens=max_tokens,
                temperature=temperature,
            )
        raise LLMProviderError("Unsupported LLM provider")

    def generate_resource_permission_sync(
        self,
        *,
        message: str,
        intelligence: dict,
        max_tokens: int,
        temperature: float,
    ) -> str:
        return self._generate_prompted_reply(
            system_prompt=load_prompt("gemma_resource_permission"),
            user_payload={"message": message, "intelligence": intelligence},
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_resource_recommendation_sync(
        self,
        *,
        message: str,
        intelligence: dict,
        resources: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        return self._generate_prompted_reply(
            system_prompt=load_prompt("gemma_resource_recommendation"),
            user_payload={"message": message, "intelligence": intelligence, "selected_catalog_records": resources},
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def classify_resource_intent_sync(self, *, message: str, max_tokens: int, temperature: float) -> dict[str, Any]:
        if self.provider == TEST_PROVIDER:
            if self.app_env != "test":
                raise LLMProviderError("Test LLM provider is only allowed when APP_ENV=test")
            return {
                "user_requested_resource": False,
                "requested_media_types": [],
                "requested_topic": None,
                "confidence": 0.0,
                "reason": "test_provider_default",
            }
        reply = self._generate_prompted_reply(
            system_prompt=load_prompt("resource_intent_classifier"),
            user_payload={"message": message},
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            import json

            parsed = json.loads(reply)
        except (TypeError, ValueError) as exc:
            raise LLMProviderError("Resource intent classifier returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise LLMProviderError("Resource intent classifier returned non-object JSON")
        return parsed

    def _generate_test_reply(self, *, message: str, intent: str, rag_context: list[dict]) -> str:
        """Deterministic test-only provider; normal local development must use openai_compatible."""
        _ = self.model
        context = " with context" if rag_context else ""
        return f"test provider reply for {intent}{context}: {message}"

    def _generate_fast_test_reply(
        self,
        *,
        message: str,
        intent_hint: str,
        clarification_may_be_needed: bool,
        style: MbtiStyle,
    ) -> str:
        _ = self.model
        _ = (clarification_may_be_needed, style)
        return f"test provider fast reply for {intent_hint}: {message}"

    def _generate_openai_compatible_simple_reply(
        self,
        *,
        message: str,
        rag_disabled: bool,
        intelligence: dict,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if not self.api_key:
            raise LLMProviderError("LLM API key is required")
        if not self.timeout_seconds:
            raise LLMProviderError("LLM timeout is required")
        try:
            from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
        except ModuleNotFoundError as exc:
            raise LLMProviderError("LLM provider package is not installed") from exc

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries or 0,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": load_prompt("gemma_simple_chat")},
                    {"role": "user", "content": self._simple_user_prompt(message=message, rag_disabled=rag_disabled, intelligence=intelligence)},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMProviderError("LLM provider returned an empty response")
            return content.strip()
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            logger.warning(
                "llm_provider_unavailable",
                provider=self.provider,
                base_url=self.base_url,
                model=self.model,
                error=str(exc),
            )
            raise LLMProviderError("LLM provider request failed") from exc
        except Exception as exc:
            logger.warning(
                "llm_provider_failed",
                provider=self.provider,
                base_url=self.base_url,
                model=self.model,
                error=str(exc),
            )
            raise LLMProviderError("LLM provider failed") from exc

    def _generate_prompted_reply(
        self,
        *,
        system_prompt: str,
        user_payload: dict,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if self.provider == TEST_PROVIDER:
            if self.app_env != "test":
                raise LLMProviderError("Test LLM provider is only allowed when APP_ENV=test")
            return f"test provider prompted reply: {user_payload.get('message', '')}"
        if self.provider not in {OPENAI_COMPATIBLE_PROVIDER, GITHUB_MODELS_PROVIDER}:
            raise LLMProviderError("Unsupported LLM provider")
        if not self.api_key:
            raise LLMProviderError("LLM API key is required")
        if not self.timeout_seconds:
            raise LLMProviderError("LLM timeout is required")
        try:
            from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
        except ModuleNotFoundError as exc:
            raise LLMProviderError("LLM provider package is not installed") from exc

        try:
            import json

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries or 0,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMProviderError("LLM provider returned an empty response")
            return content.strip()
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            logger.warning("llm_provider_unavailable", provider=self.provider, base_url=self.base_url, model=self.model, error=str(exc))
            raise LLMProviderError("LLM provider request failed") from exc
        except Exception as exc:
            logger.warning("llm_provider_failed", provider=self.provider, base_url=self.base_url, model=self.model, error=str(exc))
            raise LLMProviderError("LLM provider failed") from exc

    def _generate_openai_compatible_fast_reply(
        self,
        *,
        message: str,
        intent_hint: str,
        intensity_hint: int,
        clarification_may_be_needed: bool,
        style: MbtiStyle,
        session_summary: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if not self.api_key:
            raise LLMProviderError("LLM API key is required")
        if not self.timeout_seconds:
            raise LLMProviderError("LLM timeout is required")
        try:
            from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
        except ModuleNotFoundError as exc:
            raise LLMProviderError("LLM provider package is not installed") from exc

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=self.timeout_seconds,
                max_retries=0,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._fast_system_prompt(style=style)},
                    {
                        "role": "user",
                        "content": self._fast_user_prompt(
                            message=message,
                            intent_hint=intent_hint,
                            intensity_hint=intensity_hint,
                            clarification_may_be_needed=clarification_may_be_needed,
                            session_summary=session_summary,
                        ),
                    },
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMProviderError("LLM provider returned an empty response")
            return content.strip()
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            raise LLMProviderError("LLM provider request failed") from exc
        except Exception as exc:
            raise LLMProviderError("LLM provider failed") from exc

    def _generate_openai_compatible_reply(
        self,
        *,
        message: str,
        intent: str,
        intensity: int,
        style: MbtiStyle,
        rag_context: list[dict],
        plan: dict,
    ) -> str:
        if not self.api_key:
            raise LLMProviderError("LLM API key is required")
        if not self.timeout_seconds:
            raise LLMProviderError("LLM timeout is required")
        if self.max_retries is None:
            raise LLMProviderError("LLM retry count is required")
        try:
            from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
        except ModuleNotFoundError as exc:
            raise LLMProviderError("LLM provider package is not installed") from exc

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt(style=style)},
                    {"role": "user", "content": self._user_prompt(message, intent, intensity, rag_context, plan)},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMProviderError("LLM provider returned an empty response")
            return content.strip()
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            raise LLMProviderError("LLM provider request failed") from exc
        except Exception as exc:
            raise LLMProviderError("LLM provider failed") from exc

    def _system_prompt(self, *, style: MbtiStyle) -> str:
        return "\n\n".join(
            [
                load_prompt("gemma_langgraph_reasoning"),
                load_prompt("safety_policy"),
                "Communication style modifier only:",
                f"- tone: {style.tone}",
                f"- challenge_level: {style.challenge_level}",
                f"- answer_length: {style.answer_length}",
                f"- structure_preference: {style.structure_preference}",
                f"- avoid_patterns: {', '.join(style.avoid_patterns)}",
            ]
        )

    def _user_prompt(self, message: str, intent: str, intensity: int, rag_context: list[dict], plan: dict) -> str:
        context = "\n".join(f"- {item.get('content', '')}" for item in rag_context[:5])
        if intent == "clarification_needed":
            return (
                f"{load_prompt('clarification_response')}\n\n"
                f"style_constraints: {plan}\n"
                f"user_message:\n{message}"
            )
        return (
            f"intent: {intent}\n"
            f"intensity: {intensity}\n"
            f"plan: {plan}\n"
            f"retrieved_context:\n{context or 'none'}\n\n"
            f"user_message:\n{message}"
        )

    def _fast_system_prompt(self, *, style: MbtiStyle) -> str:
        return "\n".join(
            [
                load_prompt("gemma_fast_chat"),
                "",
                "Style hints:",
                f"tone={style.tone}",
                f"challenge={style.challenge_level}",
                f"length={style.answer_length}",
                f"structure={style.structure_preference}",
                f"avoid={', '.join(style.avoid_patterns)}",
            ]
        )

    def _fast_user_prompt(
        self,
        *,
        message: str,
        intent_hint: str,
        intensity_hint: int,
        clarification_may_be_needed: bool,
        session_summary: str,
    ) -> str:
        return (
            f"intent_hint: {intent_hint}\n"
            f"intensity_hint: {intensity_hint}\n"
            f"clarification_may_be_needed: {clarification_may_be_needed}\n"
            f"session_summary: {session_summary or 'none'}\n"
            f"user_message: {message}"
        )

    def _simple_user_prompt(self, *, message: str, rag_disabled: bool, intelligence: dict) -> str:
        import json

        return json.dumps(
            {
                "message": message,
                "metadata": intelligence,
                "document_retrieval_disabled": rag_disabled,
                "resource_flow_selected": False,
            },
            ensure_ascii=False,
        )

    async def generate_gemma_reply(self, **kwargs) -> str:
        return self.generate_gemma_reply_sync(**kwargs)

    async def stream_gemma_reply(self, **kwargs) -> AsyncIterator[str]:
        reply = await self.generate_gemma_reply(**kwargs)
        for word in reply.split():
            yield word + " "

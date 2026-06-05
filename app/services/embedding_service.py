import hashlib
import math

from app.core.config import Settings
from app.services.llm_service import OPENAI_COMPATIBLE_PROVIDER, OPENAI_PROVIDER, TEST_PROVIDER


class EmbeddingProviderError(RuntimeError):
    pass


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.settings.embedding_provider == TEST_PROVIDER:
            if self.settings.app_env != "test":
                raise EmbeddingProviderError("Test embedding provider is only allowed when APP_ENV=test")
            return [self._local_embedding(text) for text in texts]
        if self.settings.embedding_provider in {OPENAI_PROVIDER, OPENAI_COMPATIBLE_PROVIDER}:
            return self._openai_compatible_embeddings(texts)
        raise EmbeddingProviderError("Unsupported embedding provider")

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _local_embedding(self, text: str) -> list[float]:
        dimensions = self.settings.embedding_dimensions
        vector = [0.0 for _ in range(dimensions)]
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _openai_compatible_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.effective_embedding_api_key:
            raise EmbeddingProviderError("Embedding API key is required")
        if not self.settings.llm_timeout_seconds:
            raise EmbeddingProviderError("Embedding timeout is required")
        try:
            from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
        except ModuleNotFoundError as exc:
            raise EmbeddingProviderError("Embedding provider package is not installed") from exc

        try:
            client = OpenAI(
                api_key=self.settings.effective_embedding_api_key,
                base_url=self.settings.embedding_base_url or None,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
            )
            response = client.embeddings.create(model=self.settings.embedding_model, input=texts)
            return [item.embedding for item in response.data]
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            raise EmbeddingProviderError("Embedding provider request failed") from exc
        except Exception as exc:
            raise EmbeddingProviderError("Embedding provider failed") from exc

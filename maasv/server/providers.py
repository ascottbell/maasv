"""LLM and Embedding provider implementations for maasv."""

import logging

logger = logging.getLogger(__name__)


class AnthropicLLM:
    """LLM provider using Anthropic's API."""

    def __init__(self, api_key: str, default_model: str = "claude-haiku-4-5-20251001"):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._default_model = default_model

    def call(
        self,
        messages: list[dict],
        model: str = "",
        max_tokens: int = 1024,
        source: str = "",
    ) -> str:
        model = model or self._default_model
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return response.content[0].text


class OpenAILLM:
    """LLM provider using OpenAI's API."""

    def __init__(self, api_key: str, default_model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._default_model = default_model

    def call(
        self,
        messages: list[dict],
        model: str = "",
        max_tokens: int = 1024,
        source: str = "",
    ) -> str:
        model = model or self._default_model
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return response.choices[0].message.content


class VoyageEmbed:
    """Embedding provider using Voyage AI."""

    def __init__(self, api_key: str, model: str = "voyage-3-lite"):
        import voyageai
        self._client = voyageai.Client(api_key=api_key)
        self._model = model

    def embed(self, text: str) -> list[float]:
        result = self._client.embed([text], model=self._model, input_type="document")
        return result.embeddings[0]

    def embed_query(self, text: str) -> list[float]:
        result = self._client.embed([text], model=self._model, input_type="query")
        return result.embeddings[0]


class OpenAIEmbed:
    """Embedding provider using OpenAI's API."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(input=text, model=self._model)
        return response.data[0].embedding

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


def create_llm(provider: str, api_key: str, model: str):
    """Factory for LLM providers."""
    if provider == "anthropic":
        return AnthropicLLM(api_key=api_key, default_model=model)
    elif provider == "openai":
        return OpenAILLM(api_key=api_key, default_model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'anthropic' or 'openai'.")


def create_embed(provider: str, api_key: str = "", model: str = "", base_url: str = "", dims: int = 1024):
    """Factory for embedding providers."""
    if provider == "ollama":
        from maasv.providers.ollama import OllamaEmbed
        kwargs = {"dims": dims}
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        return OllamaEmbed(**kwargs)
    elif provider == "voyage":
        return VoyageEmbed(api_key=api_key, model=model)
    elif provider == "openai":
        return OpenAIEmbed(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown embed provider: {provider}. Use 'ollama', 'voyage', or 'openai'.")

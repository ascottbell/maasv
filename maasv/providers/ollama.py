"""Ollama embedding provider using Qwen3-Embedding with Matryoshka truncation."""

import json
import math
import urllib.request
from typing import Optional


class OllamaEmbed:
    """
    EmbedProvider backed by a local Ollama instance.

    Default model is qwen3-embedding:8b which supports Matryoshka
    dimensionality reduction and instruction-based query embedding.
    Vectors are truncated to `dims` and L2-normalized before return.
    """

    def __init__(
        self,
        model: str = "qwen3-embedding:8b",
        base_url: str = "http://localhost:11434",
        dims: Optional[int] = None,
        query_instruction: Optional[str] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dims = dims  # None = resolve from config at first call
        self._query_instruction = (
            query_instruction
            if query_instruction is not None
            else "Instruct: Retrieve semantically similar content.\nQuery: "
        )

    @property
    def dims(self) -> int:
        if self._dims is None:
            import maasv
            self._dims = maasv.get_config().embed_dims
        return self._dims

    def _raw_embed(self, text: str) -> list[float]:
        """Call Ollama /api/embed and return the raw vector."""
        payload = json.dumps({"model": self.model, "input": text}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        return data["embeddings"][0]

    def _truncate_and_normalize(self, vec: list[float]) -> list[float]:
        """Matryoshka truncation + L2 normalization."""
        truncated = vec[: self.dims]
        norm = math.sqrt(sum(x * x for x in truncated))
        if norm == 0:
            return truncated
        return [x / norm for x in truncated]

    def embed(self, text: str) -> list[float]:
        """Embed a document/memory for storage."""
        return self._truncate_and_normalize(self._raw_embed(text))

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with retrieval instruction prefix."""
        return self._truncate_and_normalize(
            self._raw_embed(f"{self._query_instruction}{text}")
        )

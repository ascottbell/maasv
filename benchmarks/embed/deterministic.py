"""Deterministic embedding provider for benchmarks.

Uses cluster keywords to produce embeddings with real semantic similarity:
memories/queries about the same topic cluster near each other in vector space.
"""

from __future__ import annotations

import hashlib
import math
import struct


class DeterministicEmbedProvider:
    """Keyword-cluster-based deterministic embeddings.

    Each cluster gets a random centroid (from seed). Text embedding is a weighted
    blend of matching cluster centroids + hash-based noise, L2-normalized.

    This makes vector search actually work: memories about "Atlas" cluster near
    queries about "Atlas", unlike pure hash mocks which produce random noise.
    """

    def __init__(
        self,
        dims: int = 64,
        cluster_keywords: dict[str, list[str]] | None = None,
        seed: int = 42,
        noise_weight: float = 0.10,
    ):
        self.dims = dims
        self.cluster_keywords = cluster_keywords or {}
        self.seed = seed
        self.noise_weight = noise_weight
        self._centroids: dict[str, list[float]] = {}
        self._build_centroids()

    def _build_centroids(self) -> None:
        """Generate one deterministic random centroid per cluster from seed."""
        for i, cluster_name in enumerate(sorted(self.cluster_keywords)):
            # Seed each cluster deterministically
            cluster_seed = hashlib.sha256(
                f"{self.seed}:{cluster_name}:{i}".encode()
            ).digest()
            centroid = self._bytes_to_vector(cluster_seed)
            # L2-normalize the centroid
            self._centroids[cluster_name] = self._l2_normalize(centroid)

    def _bytes_to_vector(self, data: bytes) -> list[float]:
        """Convert bytes to a float vector of self.dims length."""
        # Expand bytes if needed
        expanded = data
        while len(expanded) < self.dims * 4:
            expanded += hashlib.sha256(expanded).digest()
        # Unpack as floats in [-1, 1] range
        vec = []
        for j in range(self.dims):
            # Use 4 bytes per dimension, convert to float in [-1, 1]
            val = struct.unpack_from("B", expanded, j)[0]
            vec.append((val / 127.5) - 1.0)
        return vec

    def _l2_normalize(self, vec: list[float]) -> list[float]:
        """L2-normalize a vector. Returns zero vector if norm is 0."""
        norm = math.sqrt(sum(x * x for x in vec))
        if norm < 1e-10:
            return [0.0] * len(vec)
        return [x / norm for x in vec]

    def _hash_vector(self, text: str) -> list[float]:
        """Generate a deterministic hash-based vector for text."""
        h = hashlib.sha256(text.encode()).digest()
        return self._bytes_to_vector(h)

    def _detect_clusters(self, text: str) -> dict[str, float]:
        """Find matching clusters and their weights based on keyword presence."""
        text_lower = text.lower()
        matches: dict[str, float] = {}
        for cluster_name, keywords in self.cluster_keywords.items():
            weight = 0.0
            for kw in keywords:
                if kw.lower() in text_lower:
                    weight += 1.0
            if weight > 0:
                matches[cluster_name] = weight
        return matches

    def _embed_impl(self, text: str) -> list[float]:
        """Core embedding logic shared by embed() and embed_query()."""
        cluster_matches = self._detect_clusters(text)
        hash_vec = self._hash_vector(text)

        if not cluster_matches:
            # No cluster match: pure hash-based embedding (far from all clusters)
            return self._l2_normalize(hash_vec)

        # Compute weighted centroid from matching clusters
        total_weight = sum(cluster_matches.values())
        centroid = [0.0] * self.dims
        for cluster_name, weight in cluster_matches.items():
            cluster_centroid = self._centroids[cluster_name]
            normalized_weight = weight / total_weight
            for d in range(self.dims):
                centroid[d] += cluster_centroid[d] * normalized_weight

        # Blend: (1 - noise_weight) * centroid + noise_weight * hash_noise
        blended = [
            (1.0 - self.noise_weight) * centroid[d]
            + self.noise_weight * hash_vec[d]
            for d in range(self.dims)
        ]

        return self._l2_normalize(blended)

    def embed(self, text: str) -> list[float]:
        """Get embedding vector for a document/memory."""
        return self._embed_impl(text)

    def embed_query(self, text: str) -> list[float]:
        """Get embedding vector for a search query."""
        return self._embed_impl(text)

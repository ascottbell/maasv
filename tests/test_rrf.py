"""Tests for Reciprocal Rank Fusion, graph signal, and access frequency improvements."""

from maasv.core.retrieval import (
    _reciprocal_rank_fusion,
    _query_to_entity_fts,
    _access_frequency_bonus,
)


def _make_items(ids: list[str]) -> list[dict]:
    """Helper to create ranked lists of dicts with 'id' keys."""
    return [{"id": id_, "content": f"content for {id_}"} for id_ in ids]


class TestRRF:
    def test_basic_fusion(self):
        """RRF with single k value produces expected scores."""
        list1 = _make_items(["a", "b", "c"])
        list2 = _make_items(["b", "a", "d"])
        result = _reciprocal_rank_fusion([list1, list2], k=60)

        ids = [r["id"] for r in result]
        # "a" and "b" appear in both lists, should be top 2
        assert ids[0] in ("a", "b")
        assert ids[1] in ("a", "b")
        # All 4 unique items present
        assert set(ids) == {"a", "b", "c", "d"}

    def test_per_list_k_values(self):
        """Per-list k values change the relative weighting of signals."""
        list1 = _make_items(["a", "b"])  # vector signal
        list2 = _make_items(["b", "a"])  # bm25 signal

        # With same k for both, "a" and "b" have equal RRF scores
        # (a: 1/(k+1) + 1/(k+2), b: 1/(k+2) + 1/(k+1))
        result_same_k = _reciprocal_rank_fusion([list1, list2], k=60)
        scores_same = {r["id"]: r["rrf_score"] for r in result_same_k}
        assert abs(scores_same["a"] - scores_same["b"]) < 1e-10

        # With k_per_list=[10, 60], list1's top rank contributes more
        # because 1/(10+1) >> 1/(60+1), so "a" (rank 0 in list1) wins
        result_diff_k = _reciprocal_rank_fusion([list1, list2], k_per_list=[10, 60])
        scores_diff = {r["id"]: r["rrf_score"] for r in result_diff_k}
        assert scores_diff["a"] > scores_diff["b"]

    def test_k_per_list_fallback_to_default(self):
        """When k_per_list is shorter than ranked_lists, remaining use default k."""
        list1 = _make_items(["a"])
        list2 = _make_items(["b"])
        list3 = _make_items(["c"])

        result = _reciprocal_rank_fusion([list1, list2, list3], k=60, k_per_list=[10, 20])
        scores = {r["id"]: r["rrf_score"] for r in result}
        # list1 uses k=10, list2 uses k=20, list3 uses default k=60
        assert abs(scores["a"] - 1.0 / (10 + 0 + 1)) < 1e-10
        assert abs(scores["b"] - 1.0 / (20 + 0 + 1)) < 1e-10
        assert abs(scores["c"] - 1.0 / (60 + 0 + 1)) < 1e-10

    def test_k_per_list_none_uses_default(self):
        """When k_per_list is None, all lists use the default k."""
        list1 = _make_items(["a"])
        list2 = _make_items(["b"])

        result_none = _reciprocal_rank_fusion([list1, list2], k=30, k_per_list=None)
        result_default = _reciprocal_rank_fusion([list1, list2], k=30)

        scores_none = {r["id"]: r["rrf_score"] for r in result_none}
        scores_default = {r["id"]: r["rrf_score"] for r in result_default}
        assert scores_none == scores_default

    def test_backward_compat_default_k60(self):
        """Default behavior (no k_per_list) is identical to old k=60."""
        list1 = _make_items(["a", "b", "c"])
        list2 = _make_items(["c", "a"])

        result = _reciprocal_rank_fusion([list1, list2])
        scores = {r["id"]: r["rrf_score"] for r in result}

        # "a": rank 0 in list1 + rank 1 in list2
        expected_a = 1.0 / (60 + 0 + 1) + 1.0 / (60 + 1 + 1)
        assert abs(scores["a"] - expected_a) < 1e-10

        # "c": rank 2 in list1 + rank 0 in list2
        expected_c = 1.0 / (60 + 2 + 1) + 1.0 / (60 + 0 + 1)
        assert abs(scores["c"] - expected_c) < 1e-10

    def test_empty_lists_filtered(self):
        """Empty ranked lists contribute nothing."""
        list1 = _make_items(["a", "b"])
        result = _reciprocal_rank_fusion([list1, []], k=60)
        assert len(result) == 2
        assert result[0]["id"] == "a"

    def test_lower_k_makes_top_rank_more_dominant(self):
        """Lower k value makes the top-ranked item's score proportionally larger."""
        items = _make_items(["a", "b", "c", "d", "e"])

        result_low_k = _reciprocal_rank_fusion([items], k=10)
        result_high_k = _reciprocal_rank_fusion([items], k=100)

        # Ratio of rank-0 to rank-4 score
        low_ratio = result_low_k[0]["rrf_score"] / result_low_k[-1]["rrf_score"]
        high_ratio = result_high_k[0]["rrf_score"] / result_high_k[-1]["rrf_score"]

        # Lower k should produce a higher ratio (more top-heavy)
        assert low_ratio > high_ratio


class TestEntityFTS:
    def test_basic_or_join(self):
        """Query terms are joined with OR."""
        result = _query_to_entity_fts("MyApp architecture")
        assert "OR" in result
        assert "MyApp" in result
        assert "architecture" in result

    def test_stop_words_removed(self):
        """Common stop words are filtered out."""
        result = _query_to_entity_fts("what is the architecture of MyApp")
        assert "what" not in result.split()
        assert "the" not in result.split()
        assert "MyApp" in result

    def test_expanded_stop_words(self):
        """Extended stop words like 'how', 'does', 'about' are filtered."""
        result = _query_to_entity_fts("how does MyApp work")
        terms = [t.strip() for t in result.replace(" OR ", "|").split("|")]
        assert "how" not in terms
        assert "does" not in terms
        assert "MyApp" in terms

    def test_prefix_matching_for_long_words(self):
        """Words >= 4 chars get prefix matching added."""
        result = _query_to_entity_fts("FastAPI configuration")
        assert "FastAPI*" in result
        assert "configuration*" in result

    def test_short_words_no_prefix(self):
        """Words < 4 chars don't get prefix matching."""
        result = _query_to_entity_fts("API db setup")
        assert "API*" not in result
        assert "db*" not in result

    def test_single_word_query(self):
        """Single meaningful word produces result."""
        result = _query_to_entity_fts("FastAPI")
        assert "FastAPI" in result

    def test_all_stop_words_returns_original(self):
        """If all words are stop words, return original query."""
        result = _query_to_entity_fts("is it a")
        assert result == "is it a"

    def test_short_words_filtered(self):
        """Single-char words are filtered."""
        result = _query_to_entity_fts("a b MyApp")
        terms = [t.strip() for t in result.replace(" OR ", "|").split("|")]
        # 'a' and 'b' should not appear as terms
        assert "a" not in terms
        assert "b" not in terms


class TestAccessFrequencyBonus:
    def test_zero_access_returns_zero(self):
        """No accesses = no bonus."""
        assert _access_frequency_bonus(0, 0) == 0.0
        assert _access_frequency_bonus(0, 10) == 0.0

    def test_positive_access_returns_bonus(self):
        """Positive access count produces a bonus."""
        bonus = _access_frequency_bonus(5, 0)
        assert bonus > 0.0

    def test_more_access_higher_bonus(self):
        """More accesses = higher bonus (up to cap)."""
        low = _access_frequency_bonus(2, 0)
        high = _access_frequency_bonus(20, 0)
        assert high > low

    def test_cap_at_004(self):
        """Bonus is capped at 0.04."""
        bonus = _access_frequency_bonus(10000, 1)
        assert bonus <= 0.04

    def test_high_conversion_rate_beats_low(self):
        """High access/surfacing ratio produces higher bonus than low ratio."""
        # 10 accesses, 5 surfacings = 2.0 conversion rate
        high_conv = _access_frequency_bonus(10, 5)
        # 10 accesses, 100 surfacings = 0.1 conversion rate
        low_conv = _access_frequency_bonus(10, 100)
        assert high_conv > low_conv

    def test_negative_access_returns_zero(self):
        """Negative access count treated as zero."""
        assert _access_frequency_bonus(-5, 0) == 0.0

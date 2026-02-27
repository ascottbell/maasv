"""
Tests for the Model Router and After-Action Review modules.

Covers:
- Task classification (pattern matching for tier selection)
- Model router configuration and routing
- After-action review lifecycle (start -> steps -> complete)
- Run queries (recent, failed, by source, search)
- Run statistics and analysis
"""

import hashlib
from datetime import datetime, timezone

import pytest


# ============================================================================
# MOCK PROVIDERS
# ============================================================================


class MockEmbedProvider:
    def __init__(self, dims=64):
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = [b / 255.0 for b in h]
        while len(vec) < self.dims:
            vec.extend(vec)
        return vec[: self.dims]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


class MockLLMProvider:
    def call(self, messages, model, max_tokens, source=""):
        return "[]"


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def maasv_db(tmp_path_factory):
    """Initialize maasv with a fresh test database."""
    import maasv
    from maasv.config import MaasvConfig

    tmpdir = tmp_path_factory.mktemp("router_aar_test")
    db_path = tmpdir / "test.db"

    config = MaasvConfig(
        db_path=db_path,
        embed_dims=64,
        extraction_model="test-model",
        inference_model="test-model",
        review_model="test-model",
        cross_encoder_enabled=False,
    )

    maasv.init(config=config, llm=MockLLMProvider(), embed=MockEmbedProvider(dims=64))
    return db_path


# ============================================================================
# MODEL ROUTER — CLASSIFICATION TESTS
# ============================================================================


class TestTaskClassification:
    """Test task complexity classification."""

    def test_simple_classification_routes_local(self):
        from maasv.core.model_router import classify_task, ModelTier

        result = classify_task("Classify this email as urgent or routine")
        assert result.tier == ModelTier.LOCAL

    def test_yes_no_question_routes_local(self):
        from maasv.core.model_router import classify_task, ModelTier

        result = classify_task("Is this message spam?")
        assert result.tier == ModelTier.LOCAL

    def test_extraction_routes_haiku(self):
        from maasv.core.model_router import classify_task, ModelTier

        result = classify_task("Extract entities and relationships from this text as JSON")
        assert result.tier == ModelTier.HAIKU

    def test_summarization_routes_haiku(self):
        from maasv.core.model_router import classify_task, ModelTier

        result = classify_task("Summarize this meeting transcript")
        assert result.tier == ModelTier.HAIKU

    def test_code_generation_routes_sonnet(self):
        from maasv.core.model_router import classify_task, ModelTier

        result = classify_task("Write a Python function to parse ISO dates")
        assert result.tier == ModelTier.SONNET

    def test_complex_analysis_routes_sonnet(self):
        from maasv.core.model_router import classify_task, ModelTier

        result = classify_task("Analyze the performance characteristics of this algorithm")
        assert result.tier == ModelTier.SONNET

    def test_critical_decision_routes_opus(self):
        from maasv.core.model_router import classify_task, ModelTier

        result = classify_task("Review the security decision about authentication middleware")
        assert result.tier == ModelTier.OPUS

    def test_architecture_routes_opus(self):
        from maasv.core.model_router import classify_task, ModelTier

        result = classify_task("Design the system architecture for the new ingestion pipeline")
        assert result.tier == ModelTier.OPUS

    def test_tool_requirement_bumps_to_haiku(self):
        from maasv.core.model_router import classify_task, ModelTier

        result = classify_task("Is this urgent?", requires_tools=True)
        assert result.tier >= ModelTier.HAIKU

    def test_large_context_bumps_tier(self):
        from maasv.core.model_router import classify_task, ModelTier

        result = classify_task("Classify this", context_tokens=40_000)
        assert result.tier >= ModelTier.HAIKU

    def test_very_large_context_bumps_to_sonnet(self):
        from maasv.core.model_router import classify_task, ModelTier

        result = classify_task("Summarize this", context_tokens=120_000)
        assert result.tier >= ModelTier.SONNET

    def test_min_tier_override(self):
        from maasv.core.model_router import classify_task, ModelTier

        result = classify_task("Is this spam?", min_tier=ModelTier.SONNET)
        assert result.tier == ModelTier.SONNET
        assert "override" in result.reason.lower()


# ============================================================================
# MODEL ROUTER — ROUTER CLASS TESTS
# ============================================================================


class TestModelRouter:
    """Test the ModelRouter class."""

    def test_default_router_has_all_tiers(self):
        from maasv.core.model_router import ModelRouter, ModelTier

        router = ModelRouter()
        for tier in ModelTier:
            config = router.get_tier(tier)
            assert config.model_id != ""

    def test_route_returns_tier_config(self):
        from maasv.core.model_router import ModelRouter

        router = ModelRouter()
        config = router.route("Classify this event as urgent or routine")
        assert config.is_local  # should go to local for classification

    def test_route_with_reason(self):
        from maasv.core.model_router import ModelRouter

        router = ModelRouter()
        config, classification = router.route_with_reason("Write code for a REST API")
        assert classification.reason != ""
        assert classification.tier.name in ("SONNET", "OPUS", "HAIKU")

    def test_set_model_override(self):
        from maasv.core.model_router import ModelRouter, ModelTier

        router = ModelRouter()
        router.set_model(ModelTier.LOCAL, "custom-8b:latest")
        assert router.get_model_id(ModelTier.LOCAL) == "custom-8b:latest"

    def test_get_stats(self):
        from maasv.core.model_router import ModelRouter

        router = ModelRouter()
        stats = router.get_stats()
        assert "LOCAL" in stats
        assert "HAIKU" in stats
        assert "SONNET" in stats
        assert "OPUS" in stats

    def test_module_level_route(self):
        from maasv.core.model_router import route

        config = route("Is this email important?")
        assert config is not None
        assert config.model_id != ""


# ============================================================================
# AFTER-ACTION REVIEW — LIFECYCLE TESTS
# ============================================================================


class TestRunLifecycle:
    """Test run record create/update/complete lifecycle."""

    def test_start_run(self, maasv_db):
        from maasv.core.after_action import start_run, get_run, RunOutcome

        run_id = start_run(
            goal="Send morning briefing to Adam",
            source="proactive",
            trigger="scheduled_7am",
            model_used="claude-haiku-4-5-20251001",
            model_tier="haiku",
        )
        assert run_id.startswith("run_")

        run = get_run(run_id)
        assert run is not None
        assert run.goal == "Send morning briefing to Adam"
        assert run.outcome == RunOutcome.PENDING
        assert run.source == "proactive"

    def test_add_steps(self, maasv_db):
        from maasv.core.after_action import start_run, add_step, get_run

        run_id = start_run(goal="Check weather and calendar")

        add_step(run_id, "get_weather", "location=NYC", "Sunny, 65F", True, duration_ms=150)
        add_step(run_id, "get_calendar_events", "date=today", "3 events found", True, duration_ms=200)
        add_step(run_id, "format_briefing", "weather+calendar", "Briefing formatted", True, duration_ms=50)

        run = get_run(run_id)
        assert len(run.steps) == 3
        assert run.steps[0].tool_name == "get_weather"
        assert run.steps[1].tool_name == "get_calendar_events"
        assert run.steps[2].tool_name == "format_briefing"
        assert all(s.success for s in run.steps)

    def test_add_failed_step(self, maasv_db):
        from maasv.core.after_action import start_run, add_step, get_run

        run_id = start_run(goal="Send email to John")

        add_step(run_id, "send_email", "to=john@example.com", "", False,
                 error="SMTP connection refused")

        run = get_run(run_id)
        assert len(run.steps) == 1
        assert not run.steps[0].success
        assert "SMTP" in run.steps[0].error

    def test_complete_run_success(self, maasv_db):
        from maasv.core.after_action import start_run, add_step, complete_run, get_run, RunOutcome

        run_id = start_run(goal="Complete task successfully")
        add_step(run_id, "some_tool", "args", "result", True)
        complete_run(run_id, RunOutcome.SUCCESS, "All done", input_tokens=500, output_tokens=200)

        run = get_run(run_id)
        assert run.outcome == RunOutcome.SUCCESS
        assert run.completed_at is not None
        assert run.input_tokens == 500
        assert run.output_tokens == 200

    def test_complete_run_failure(self, maasv_db):
        from maasv.core.after_action import start_run, complete_run, get_run, RunOutcome

        run_id = start_run(goal="Task that will fail")
        complete_run(run_id, RunOutcome.FAILURE, "Could not connect to API")

        run = get_run(run_id)
        assert run.outcome == RunOutcome.FAILURE
        assert "Could not connect" in run.outcome_details

    def test_cannot_complete_already_completed(self, maasv_db):
        from maasv.core.after_action import start_run, complete_run, RunOutcome

        run_id = start_run(goal="Already done")
        complete_run(run_id, RunOutcome.SUCCESS, "Done")

        # Second completion should return False (already completed)
        result = complete_run(run_id, RunOutcome.FAILURE, "Trying again")
        assert not result

    def test_add_feedback(self, maasv_db):
        from maasv.core.after_action import start_run, complete_run, add_feedback, get_run, RunOutcome

        run_id = start_run(goal="Task needing feedback")
        complete_run(run_id, RunOutcome.SUCCESS)

        add_feedback(run_id, score=5, notes="Great job, exactly what I needed")
        run = get_run(run_id)
        assert run.user_feedback_score == 5
        assert "Great job" in run.user_feedback_notes

    def test_feedback_score_validation(self, maasv_db):
        from maasv.core.after_action import start_run, add_feedback

        run_id = start_run(goal="Bad score test")
        with pytest.raises(ValueError, match="between 1 and 5"):
            add_feedback(run_id, score=0)

    def test_get_nonexistent_run(self, maasv_db):
        from maasv.core.after_action import get_run

        assert get_run("run_doesnotexist") is None

    def test_add_step_to_nonexistent_run(self, maasv_db):
        from maasv.core.after_action import add_step

        result = add_step("run_doesnotexist", "tool", "args", "result", True)
        assert not result


# ============================================================================
# AFTER-ACTION REVIEW — QUERY TESTS
# ============================================================================


class TestRunQueries:
    """Test run query operations."""

    def test_get_recent_runs(self, maasv_db):
        from maasv.core.after_action import get_recent_runs

        runs = get_recent_runs()
        assert isinstance(runs, list)
        assert len(runs) > 0  # we created several in lifecycle tests

    def test_get_recent_by_outcome(self, maasv_db):
        from maasv.core.after_action import get_recent_runs, RunOutcome

        successes = get_recent_runs(outcome=RunOutcome.SUCCESS)
        assert all(r.outcome == RunOutcome.SUCCESS for r in successes)

    def test_get_failed_runs(self, maasv_db):
        from maasv.core.after_action import get_failed_runs

        failures = get_failed_runs()
        assert all(r.outcome.value == "failure" for r in failures)

    def test_get_runs_by_source(self, maasv_db):
        from maasv.core.after_action import get_runs_by_source

        proactive = get_runs_by_source("proactive")
        assert all(r.source == "proactive" for r in proactive)

    def test_search_runs(self, maasv_db):
        from maasv.core.after_action import search_runs

        results = search_runs("briefing")
        assert isinstance(results, list)

    def test_get_pending_runs(self, maasv_db):
        from maasv.core.after_action import get_pending_runs, RunOutcome

        pending = get_pending_runs()
        assert all(r.outcome == RunOutcome.PENDING for r in pending)


# ============================================================================
# AFTER-ACTION REVIEW — ANALYSIS TESTS
# ============================================================================


class TestRunAnalysis:
    """Test run statistics and analysis."""

    def test_get_stats(self, maasv_db):
        from maasv.core.after_action import get_stats

        stats = get_stats()
        assert "total_runs" in stats
        assert stats["total_runs"] > 0
        assert "by_outcome" in stats
        assert "by_source" in stats
        assert "avg_steps_per_run" in stats

    def test_get_tool_success_rates(self, maasv_db):
        from maasv.core.after_action import get_tool_success_rates

        rates = get_tool_success_rates()
        assert isinstance(rates, dict)
        # We added steps with various tools in lifecycle tests
        if rates:
            for tool, stats in rates.items():
                assert "total" in stats
                assert "success_rate" in stats
                assert 0 <= stats["success_rate"] <= 1

    def test_get_failure_patterns(self, maasv_db):
        from maasv.core.after_action import get_failure_patterns

        patterns = get_failure_patterns()
        assert isinstance(patterns, list)

    def test_format_run_for_prompt(self, maasv_db):
        from maasv.core.after_action import (
            start_run, add_step, complete_run, get_run,
            RunOutcome, format_run_for_prompt,
        )

        run_id = start_run(goal="Formatting test run")
        add_step(run_id, "tool_a", "doing stuff", "stuff done", True)
        add_step(run_id, "tool_b", "more stuff", "", False, error="Connection error")
        complete_run(run_id, RunOutcome.PARTIAL, "tool_b failed but tool_a succeeded")

        run = get_run(run_id)
        formatted = format_run_for_prompt(run)
        assert "Formatting test run" in formatted
        assert "tool_a" in formatted
        assert "FAIL" in formatted

    def test_format_stats_for_prompt(self, maasv_db):
        from maasv.core.after_action import get_stats, format_stats_for_prompt

        stats = get_stats()
        formatted = format_stats_for_prompt(stats)
        assert "Task Execution History" in formatted
        assert "Total runs" in formatted


# ============================================================================
# AFTER-ACTION REVIEW — METADATA TESTS
# ============================================================================


class TestRunMetadata:
    """Test run records with metadata and context snapshots."""

    def test_run_with_metadata(self, maasv_db):
        from maasv.core.after_action import start_run, get_run

        run_id = start_run(
            goal="Task with metadata",
            metadata={"conversation_id": "conv_123", "user_intent": "schedule_meeting"},
        )

        run = get_run(run_id)
        assert run.metadata is not None
        assert run.metadata["conversation_id"] == "conv_123"

    def test_run_with_context_snapshot(self, maasv_db):
        from maasv.core.after_action import start_run, get_run

        context = "Active projects: Doris (active), TerryAnn (production). Adam is working."
        run_id = start_run(goal="Context snapshot test", context_snapshot=context)

        run = get_run(run_id)
        assert "Doris" in run.context_snapshot

    def test_step_with_wisdom_link(self, maasv_db):
        from maasv.core.after_action import start_run, add_step, get_run

        run_id = start_run(goal="Wisdom-linked step test")
        add_step(
            run_id, "send_email", "to=john@test.com",
            "Email sent", True,
            wisdom_id="wisdom_abc123",
        )

        run = get_run(run_id)
        assert run.steps[0].wisdom_id == "wisdom_abc123"

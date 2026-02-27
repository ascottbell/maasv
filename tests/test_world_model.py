"""
Tests for the World Model and Commitments modules.

Covers:
- World model entity schemas and metadata
- Confidence tracking and staleness detection
- Relevance decay computation
- Activity hypothesis
- Bi-temporal queries
- Commitment lifecycle (create -> in_progress -> completed)
- Commitment queries (active, overdue, by owner, escalations)
- Commitment search
"""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ============================================================================
# MOCK PROVIDERS (same pattern as test_decomposition.py)
# ============================================================================


class MockEmbedProvider:
    """Deterministic embeddings for testing."""

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

    tmpdir = tmp_path_factory.mktemp("world_model_test")
    db_path = tmpdir / "test.db"

    config = MaasvConfig(
        db_path=db_path,
        embed_dims=64,
        extraction_model="test-model",
        inference_model="test-model",
        review_model="test-model",
        cross_encoder_enabled=False,
        decay_half_life_days=30,
    )

    maasv.init(config=config, llm=MockLLMProvider(), embed=MockEmbedProvider(dims=64))
    return db_path


# ============================================================================
# DECAY AND CONFIDENCE TESTS
# ============================================================================


class TestDecay:
    """Test relevance decay computation."""

    def test_brand_new_fact_has_no_decay(self):
        from maasv.core.world_model import compute_decay

        now = datetime.now(timezone.utc)
        factor = compute_decay(now.isoformat(), half_life_days=30, now=now)
        assert factor == 1.0

    def test_fact_at_half_life_is_half(self):
        from maasv.core.world_model import compute_decay

        now = datetime.now(timezone.utc)
        thirty_days_ago = (now - timedelta(days=30)).isoformat()
        factor = compute_decay(thirty_days_ago, half_life_days=30, now=now)
        assert abs(factor - 0.5) < 0.01

    def test_very_old_fact_decays_heavily(self):
        from maasv.core.world_model import compute_decay

        now = datetime.now(timezone.utc)
        year_ago = (now - timedelta(days=365)).isoformat()
        factor = compute_decay(year_ago, half_life_days=30, now=now)
        assert factor < 0.001

    def test_zero_half_life_returns_zero(self):
        from maasv.core.world_model import compute_decay

        now = datetime.now(timezone.utc)
        yesterday = (now - timedelta(days=1)).isoformat()
        factor = compute_decay(yesterday, half_life_days=0, now=now)
        assert factor == 0.0

    def test_invalid_timestamp_returns_default(self):
        from maasv.core.world_model import compute_decay

        factor = compute_decay("not-a-date", half_life_days=30)
        assert factor == 0.5

    def test_effective_confidence_combines_base_and_decay(self):
        from maasv.core.world_model import get_effective_confidence

        now = datetime.now(timezone.utc)
        thirty_days_ago = (now - timedelta(days=30)).isoformat()

        # Base confidence 0.8, 30 days old with 30-day half life
        eff = get_effective_confidence(0.8, thirty_days_ago, half_life_days=30, now=now)
        assert abs(eff - 0.4) < 0.01  # 0.8 * 0.5

    def test_effective_confidence_new_fact(self):
        from maasv.core.world_model import get_effective_confidence

        now = datetime.now(timezone.utc)
        eff = get_effective_confidence(1.0, now.isoformat(), half_life_days=30, now=now)
        assert eff == 1.0


# ============================================================================
# WORLD MODEL ENTITY TESTS
# ============================================================================


class TestWorldModelEntities:
    """Test entity creation and metadata operations."""

    def test_create_person_entity(self, maasv_db):
        from maasv.core.graph import find_or_create_entity
        from maasv.core.world_model import update_entity_metadata, get_entity_state

        entity_id = find_or_create_entity("Test Person", "person", metadata={
            "relationship_to_adam": "friend",
            "interaction_frequency": "weekly",
        })
        assert entity_id.startswith("ent_")

        # Update metadata
        updated = update_entity_metadata(entity_id, {
            "last_interaction": datetime.now(timezone.utc).isoformat(),
            "roles": ["friend", "colleague"],
        })
        assert updated

        # Get state
        state = get_entity_state(entity_id)
        assert state is not None
        assert state.name == "Test Person"
        assert state.entity_type == "person"
        assert "roles" in state.metadata

    def test_create_project_entity(self, maasv_db):
        from maasv.core.graph import find_or_create_entity
        from maasv.core.world_model import set_project_state, get_projects

        entity_id = find_or_create_entity("TestProject", "project", metadata={
            "state": "active",
            "tech_stack": ["python", "sqlite"],
        })
        assert entity_id.startswith("ent_")

        # Set state
        result = set_project_state("TestProject", "blocked")
        assert result

        # Verify
        from maasv.core.graph import get_entity
        entity = get_entity(entity_id)
        # get_entity already parses metadata JSON into a dict
        meta = entity.get("metadata") or {}
        assert meta.get("state") == "blocked"

    def test_set_invalid_project_state_raises(self, maasv_db):
        from maasv.core.world_model import set_project_state

        with pytest.raises(ValueError, match="Invalid project state"):
            set_project_state("TestProject", "flying")

    def test_create_location_entity(self, maasv_db):
        from maasv.core.graph import find_or_create_entity

        entity_id = find_or_create_entity("Home UWS", "location", metadata={
            "address": "Upper West Side, Manhattan",
            "type": "home",
            "typical_occupants": ["Adam", "Gabby", "Levi", "Dani", "Billi"],
        })
        assert entity_id.startswith("ent_")

    def test_create_routine_entity(self, maasv_db):
        from maasv.core.graph import find_or_create_entity

        entity_id = find_or_create_entity("Morning Standup", "routine", metadata={
            "schedule": "weekdays 9:30am",
            "time_of_day": "morning",
            "day_pattern": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            "duration_minutes": 15,
        })
        assert entity_id.startswith("ent_")

    def test_create_device_entity(self, maasv_db):
        from maasv.core.graph import find_or_create_entity
        from maasv.core.world_model import update_device_state

        entity_id = find_or_create_entity("Mac Mini M4", "device", metadata={
            "device_type": "server",
            "capabilities": ["compute", "voice", "homeassistant"],
            "state": "online",
        })
        assert entity_id.startswith("ent_")

        result = update_device_state("Mac Mini M4", "maintenance")
        assert result

    def test_record_interaction(self, maasv_db):
        from maasv.core.world_model import record_interaction

        result = record_interaction("Test Person")
        assert result

    def test_nonexistent_entity_returns_none(self, maasv_db):
        from maasv.core.world_model import get_entity_state

        state = get_entity_state("ent_doesnotexist")
        assert state is None

    def test_update_nonexistent_entity_returns_false(self, maasv_db):
        from maasv.core.world_model import update_entity_metadata

        result = update_entity_metadata("ent_doesnotexist", {"key": "value"})
        assert result is False


# ============================================================================
# STALENESS TESTS
# ============================================================================


class TestStaleness:
    """Test staleness detection."""

    def test_get_stale_entities(self, maasv_db):
        from maasv.core.world_model import get_stale_entities

        # With a fresh DB, no entities should be very stale yet
        # but we can still call it without error
        stale = get_stale_entities(threshold_days=0)
        # threshold_days=0 means everything is stale
        assert isinstance(stale, list)

    def test_get_stale_facts(self, maasv_db):
        from maasv.core.world_model import get_stale_facts

        stale = get_stale_facts(threshold_days=0, min_confidence=0.99)
        assert isinstance(stale, list)


# ============================================================================
# ACTIVITY HYPOTHESIS TESTS
# ============================================================================


class TestActivityHypothesis:
    """Test activity estimation."""

    def test_sleeping_at_3am(self, maasv_db):
        from maasv.core.world_model import get_current_activity

        activity = get_current_activity(current_hour=3, day_of_week=2)
        assert activity.activity == "sleeping"
        assert activity.confidence >= 0.8
        assert activity.location == "home"

    def test_working_at_10am_weekday(self, maasv_db):
        from maasv.core.world_model import get_current_activity

        activity = get_current_activity(current_hour=10, day_of_week=1)
        # Could be "working" or a matched routine like "Morning Standup"
        assert "work" in activity.activity.lower() or "routine" in activity.activity.lower()
        assert activity.confidence >= 0.7

    def test_family_time_weekend(self, maasv_db):
        from maasv.core.world_model import get_current_activity

        activity = get_current_activity(current_hour=10, day_of_week=5)
        assert "family" in activity.activity.lower() or "personal" in activity.activity.lower()

    def test_evening_wind_down(self, maasv_db):
        from maasv.core.world_model import get_current_activity

        activity = get_current_activity(current_hour=21, day_of_week=3)
        assert "wind" in activity.activity.lower() or "bedtime" in activity.activity.lower()

    def test_activity_has_evidence(self, maasv_db):
        from maasv.core.world_model import get_current_activity

        activity = get_current_activity(current_hour=14, day_of_week=0)
        assert len(activity.based_on) > 0
        assert activity.reasoning != ""


# ============================================================================
# BI-TEMPORAL QUERY TESTS
# ============================================================================


class TestBitemporalQueries:
    """Test time-travel queries."""

    def test_time_travel_current(self, maasv_db):
        from maasv.core.world_model import time_travel_query

        # Without as_of, should return current state
        results = time_travel_query(entity_type="person")
        assert isinstance(results, list)

    def test_time_travel_past(self, maasv_db):
        from maasv.core.world_model import time_travel_query

        # Query the past — should not fail
        past = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        results = time_travel_query(as_of=past)
        assert isinstance(results, list)

    def test_time_travel_with_predicate_filter(self, maasv_db):
        from maasv.core.world_model import time_travel_query

        results = time_travel_query(predicate="works_on")
        assert isinstance(results, list)


# ============================================================================
# WORLD SUMMARY TESTS
# ============================================================================


class TestWorldSummary:
    """Test world state summary generation."""

    def test_summary_generates(self, maasv_db):
        from maasv.core.world_model import get_world_summary

        summary = get_world_summary()
        assert "World State" in summary
        assert "Current activity estimate" in summary

    def test_summary_with_stale(self, maasv_db):
        from maasv.core.world_model import get_world_summary

        summary = get_world_summary(include_stale=True)
        assert "World State" in summary


# ============================================================================
# COMMITMENT LIFECYCLE TESTS
# ============================================================================


class TestCommitmentLifecycle:
    """Test commitment CRUD and state transitions."""

    def test_create_commitment(self, maasv_db):
        from maasv.core.commitments import Commitment, CommitmentType, CommitmentStatus, create, get

        c = Commitment(
            commitment_type=CommitmentType.PROMISE,
            owner="adam",
            subject="Reply to John's email about the contract",
            next_action="Draft reply and send",
            verification_rule="Email sent and confirmed",
            source="conversation",
        )
        cid = create(c)
        assert cid == c.id

        # Retrieve
        fetched = get(cid)
        assert fetched is not None
        assert fetched.subject == "Reply to John's email about the contract"
        assert fetched.status == CommitmentStatus.OPEN
        assert fetched.commitment_type == CommitmentType.PROMISE

    def test_update_status_to_in_progress(self, maasv_db):
        from maasv.core.commitments import (
            Commitment, CommitmentType, CommitmentStatus,
            create, get, update_status,
        )

        c = Commitment(
            commitment_type=CommitmentType.FOLLOW_UP,
            owner="adam",
            subject="Follow up with dentist",
        )
        cid = create(c)

        updated = update_status(cid, CommitmentStatus.IN_PROGRESS, "Started calling")
        assert updated

        fetched = get(cid)
        assert fetched.status == CommitmentStatus.IN_PROGRESS
        assert "Started calling" in fetched.context

    def test_complete_commitment_sets_completed_at(self, maasv_db):
        from maasv.core.commitments import (
            Commitment, CommitmentType, CommitmentStatus,
            create, get, update_status,
        )

        c = Commitment(
            commitment_type=CommitmentType.DELEGATION,
            owner="doris",
            subject="Send birthday reminder to Gabby",
        )
        cid = create(c)

        update_status(cid, CommitmentStatus.COMPLETED, "Reminder sent")
        fetched = get(cid)
        assert fetched.status == CommitmentStatus.COMPLETED
        assert fetched.completed_at is not None

    def test_delete_commitment(self, maasv_db):
        from maasv.core.commitments import (
            Commitment, CommitmentType, create, get, delete,
        )

        c = Commitment(
            commitment_type=CommitmentType.REMINDER,
            owner="adam",
            subject="Test deletion",
        )
        cid = create(c)
        assert get(cid) is not None

        deleted = delete(cid)
        assert deleted
        assert get(cid) is None

    def test_update_next_action(self, maasv_db):
        from maasv.core.commitments import (
            Commitment, CommitmentType, create, get, update_next_action,
        )

        c = Commitment(
            commitment_type=CommitmentType.PROMISE,
            owner="adam",
            subject="Ship feature X",
            next_action="Write tests",
        )
        cid = create(c)

        update_next_action(cid, "Deploy to staging")
        fetched = get(cid)
        assert fetched.next_action == "Deploy to staging"


# ============================================================================
# COMMITMENT QUERY TESTS
# ============================================================================


class TestCommitmentQueries:
    """Test commitment query operations."""

    def test_get_active(self, maasv_db):
        from maasv.core.commitments import get_active

        active = get_active()
        assert isinstance(active, list)
        # All should have non-terminal status
        for c in active:
            assert c.status.value in ("open", "in_progress", "blocked")

    def test_get_overdue(self, maasv_db):
        from maasv.core.commitments import (
            Commitment, CommitmentType, DeadlineType,
            create, get_overdue,
        )

        # Create a commitment with a past deadline
        past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        c = Commitment(
            commitment_type=CommitmentType.PROMISE,
            owner="adam",
            subject="Overdue test item",
            deadline=past,
            deadline_type=DeadlineType.HARD,
        )
        create(c)

        overdue = get_overdue()
        assert len(overdue) >= 1
        subjects = [x.subject for x in overdue]
        assert "Overdue test item" in subjects

    def test_get_upcoming(self, maasv_db):
        from maasv.core.commitments import (
            Commitment, CommitmentType, DeadlineType,
            create, get_upcoming,
        )

        # Create a commitment with a deadline in 12 hours
        future = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
        c = Commitment(
            commitment_type=CommitmentType.FOLLOW_UP,
            owner="adam",
            subject="Upcoming test item",
            deadline=future,
            deadline_type=DeadlineType.SOFT,
        )
        create(c)

        upcoming = get_upcoming(hours=24)
        subjects = [x.subject for x in upcoming]
        assert "Upcoming test item" in subjects

    def test_get_by_owner(self, maasv_db):
        from maasv.core.commitments import get_by_owner

        adam_commitments = get_by_owner("adam")
        assert isinstance(adam_commitments, list)
        for c in adam_commitments:
            assert c.owner == "adam"

    def test_get_by_type(self, maasv_db):
        from maasv.core.commitments import CommitmentType, get_by_type

        promises = get_by_type(CommitmentType.PROMISE)
        assert isinstance(promises, list)
        for c in promises:
            assert c.commitment_type == CommitmentType.PROMISE

    def test_check_escalations(self, maasv_db):
        from maasv.core.commitments import (
            Commitment, CommitmentType, DeadlineType,
            create, check_escalations,
        )

        past = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        c = Commitment(
            commitment_type=CommitmentType.PROMISE,
            owner="adam",
            subject="Escalation test item",
            deadline=past,
            deadline_type=DeadlineType.HARD,
            escalation_policy="Notify Adam immediately, then Gabby if no response in 1h",
        )
        create(c)

        escalations = check_escalations()
        subjects = [x.subject for x in escalations]
        assert "Escalation test item" in subjects

    def test_get_waiting_on(self, maasv_db):
        from maasv.core.commitments import (
            Commitment, CommitmentType, create, get_waiting_on,
        )

        c = Commitment(
            commitment_type=CommitmentType.WAITING_ON,
            owner="adam",
            subject="Waiting for John to send proposal",
        )
        create(c)

        waiting = get_waiting_on()
        subjects = [x.subject for x in waiting]
        assert "Waiting for John to send proposal" in subjects

    def test_search_commitments(self, maasv_db):
        from maasv.core.commitments import search

        results = search("email")
        assert isinstance(results, list)
        # We created one about "Reply to John's email" earlier
        if results:
            assert any("email" in r.subject.lower() for r in results)

    def test_get_stats(self, maasv_db):
        from maasv.core.commitments import get_stats

        stats = get_stats()
        assert "total" in stats
        assert "active" in stats
        assert "overdue" in stats
        assert "by_type" in stats
        assert "by_owner" in stats
        assert stats["total"] > 0

    def test_get_nonexistent_commitment(self, maasv_db):
        from maasv.core.commitments import get

        result = get("cmt_doesnotexist")
        assert result is None


# ============================================================================
# COMMITMENT FORMATTING TESTS
# ============================================================================


class TestCommitmentFormatting:
    """Test commitment prompt formatting."""

    def test_format_empty_list(self):
        from maasv.core.commitments import format_for_prompt

        result = format_for_prompt([])
        assert result == ""

    def test_format_with_overdue(self, maasv_db):
        from maasv.core.commitments import (
            Commitment, CommitmentType, CommitmentStatus,
            DeadlineType, format_for_prompt,
        )

        past = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        c = Commitment(
            commitment_type=CommitmentType.PROMISE,
            owner="adam",
            subject="Overdue formatting test",
            status=CommitmentStatus.OPEN,
            deadline=past,
            deadline_type=DeadlineType.HARD,
            next_action="Do the thing",
        )

        result = format_for_prompt([c])
        assert "OVERDUE" in result
        assert "Do the thing" in result

    def test_format_with_upcoming(self, maasv_db):
        from maasv.core.commitments import (
            Commitment, CommitmentType, CommitmentStatus,
            DeadlineType, format_for_prompt,
        )

        future = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
        c = Commitment(
            commitment_type=CommitmentType.FOLLOW_UP,
            owner="adam",
            subject="Upcoming formatting test",
            status=CommitmentStatus.IN_PROGRESS,
            deadline=future,
            deadline_type=DeadlineType.SOFT,
        )

        result = format_for_prompt([c])
        assert "due in" in result
        assert "[~]" in result


# ============================================================================
# COMMITMENT WITH RELATED ENTITIES
# ============================================================================


class TestCommitmentEntityRelations:
    """Test commitments linked to KG entities."""

    def test_commitment_with_entity_ids(self, maasv_db):
        from maasv.core.commitments import Commitment, CommitmentType, create, get
        from maasv.core.graph import find_or_create_entity

        person_id = find_or_create_entity("John Doe", "person")
        project_id = find_or_create_entity("ContractReview", "project")

        c = Commitment(
            commitment_type=CommitmentType.PROMISE,
            owner="adam",
            subject="Review contract and send to John",
            related_entity_ids=[person_id, project_id],
            context="Discussed in meeting on Friday",
        )
        cid = create(c)

        fetched = get(cid)
        assert fetched is not None
        assert len(fetched.related_entity_ids) == 2
        assert person_id in fetched.related_entity_ids
        assert project_id in fetched.related_entity_ids

    def test_commitment_with_metadata(self, maasv_db):
        from maasv.core.commitments import Commitment, CommitmentType, create, get

        c = Commitment(
            commitment_type=CommitmentType.DRAFT_TO_SEND,
            owner="adam",
            subject="Send the proposal draft",
            metadata={"email_draft_id": "draft_abc123", "recipient": "john@example.com"},
        )
        cid = create(c)

        fetched = get(cid)
        assert fetched.metadata is not None
        assert fetched.metadata["email_draft_id"] == "draft_abc123"

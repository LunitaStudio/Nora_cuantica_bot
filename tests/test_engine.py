import hashlib
from datetime import UTC, datetime

import pytest

from nora_quantica.domain.config import (
    InterestWeights,
    StateEngineConfig,
    state_engine_config_from_dict,
)
from nora_quantica.domain.engine import StateEngine
from nora_quantica.domain.entropy import EntropySample, initial_state_from_entropy
from nora_quantica.domain.models import EntropyMetadata, MessageImpact


def make_state():
    raw = bytes([184, 112, 97, 171, 191, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    return initial_state_from_entropy(
        EntropySample(
            raw_bytes=raw,
            metadata=EntropyMetadata(
                provider="test",
                retrieved_at=datetime.now(UTC),
                raw_bytes_hash=hashlib.sha256(raw).hexdigest(),
            ),
        ),
        conversation_id="test-conversation",
    )


def make_impact(**overrides: object) -> MessageImpact:
    values = {
        "topic_relevance": 78,
        "novelty": 64,
        "continuity": 70,
        "disagreement_strength": 55,
        "social_signal": 42,
        "urgency": 12,
        "event_intensity": 18,
        "engagement_request": 35,
        "topic_shift": 10,
        "primary_topic": "philosophy_ideas",
        "secondary_topic": "technology_science",
    }
    values.update(overrides)
    return MessageImpact(**values)


def test_interest_uses_centralized_weights() -> None:
    state = make_state()
    engine = StateEngine()
    affinity = (
        0.75 * state.baseline.topic_affinities["philosophy_ideas"]
        + 0.25 * state.baseline.topic_affinities["technology_science"]
    )
    expected = 0.40 * state.baseline.interest_bias + 0.45 * affinity + 0.15 * 64
    assert engine.calculate_interest(state, make_impact()) == pytest.approx(expected)


def test_relevance_continuity_and_closeness_do_not_change_interest() -> None:
    state = make_state()
    engine = StateEngine()
    low = engine.calculate_interest(
        state,
        make_impact(topic_relevance=0, continuity=0),
    )
    high = engine.calculate_interest(
        state,
        make_impact(topic_relevance=100, continuity=100),
    )
    assert high == pytest.approx(low)


def test_topic_category_changes_interest_using_fixed_affinities() -> None:
    state = make_state()
    engine = StateEngine()
    low = engine.calculate_interest(
        state,
        make_impact(primary_topic="everyday_life", secondary_topic="none"),
    )
    high = engine.calculate_interest(
        state,
        make_impact(primary_topic="travel_nature_weather", secondary_topic="none"),
    )
    assert high > low


def test_no_topic_preserves_current_interest_with_affinity_formula() -> None:
    state = make_state()
    state.current = type(state.current)(
        availability=state.current.availability,
        current_interest=37.5,
        candor=state.current.candor,
        topic_orientation=state.current.topic_orientation,
    )
    calculated = StateEngine().calculate_interest(
        state,
        make_impact(primary_topic="none", secondary_topic="none", novelty=100),
    )
    assert calculated == 37.5


def test_normal_transition_is_bounded_and_keeps_baseline_immutable() -> None:
    state = make_state()
    baseline = state.baseline
    transition = StateEngine().apply(state, make_impact())
    assert transition.deltas.max_delta == 8
    assert all(abs(delta) <= 8 for delta in transition.deltas.effective.values())
    assert state.baseline == baseline
    assert state.turn_number == 1


def test_high_impact_selects_larger_cap() -> None:
    state = make_state()
    transition = StateEngine().apply(state, make_impact(event_intensity=90))
    assert transition.deltas.max_delta == 25


def test_high_impact_can_exceed_normal_delta_without_exceeding_its_cap() -> None:
    extreme = make_impact(
        event_intensity=90,
        continuity=0,
        topic_relevance=0,
        topic_shift=100,
        social_signal=100,
    )
    high_transition = StateEngine().apply(make_state(), extreme)
    normal_transition = StateEngine().apply(
        make_state(),
        make_impact(
            event_intensity=20,
            continuity=0,
            topic_relevance=0,
            topic_shift=100,
            social_signal=100,
        ),
    )
    assert high_transition.deltas.effective["topic_orientation"] == -10
    assert normal_transition.deltas.effective["topic_orientation"] == -8


def test_decimal_recovery_has_no_rounding_dead_zone() -> None:
    state = make_state()
    state.current = type(state.current)(
        availability=state.baseline.availability - 1,
        current_interest=state.current.current_interest,
        candor=state.current.candor,
        topic_orientation=state.current.topic_orientation,
    )
    transition = StateEngine().apply(
        state,
        make_impact(engagement_request=0, event_intensity=0),
    )
    assert transition.deltas.requested["availability"] == pytest.approx(0.05)


def test_interest_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        InterestWeights(topic_relevance=0.50)

    config = StateEngineConfig(interest_weights=InterestWeights())
    assert config.interest_weights.interest_bias == 0.40
    assert config.interest_weights.topic_affinity == 0.45


def test_legacy_interest_config_can_be_reconstructed_for_replay() -> None:
    config = state_engine_config_from_dict(
        {
            "normal_max_delta": 8,
            "high_impact_max_delta": 25,
            "high_impact_threshold": 75,
            "availability_recovery_rate": 0.05,
            "interest_weights": {
                "interest_bias": 0.30,
                "topic_relevance": 0.30,
                "novelty": 0.15,
                "continuity": 0.15,
                "relational_closeness": 0.10,
            },
        }
    )
    assert config.interest_weights.topic_affinity == 0
    assert config.interest_weights.topic_relevance == 0.30
    assert config.preserve_interest_without_topic is False
    state = make_state()
    calculated = StateEngine(config).calculate_interest(
        state,
        make_impact(primary_topic="none", secondary_topic="none"),
    )
    assert calculated != state.current.current_interest

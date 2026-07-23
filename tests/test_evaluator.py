import json

import pytest

from nora_quantica.domain.evaluator import (
    EVALUATOR_SYSTEM_PROMPT,
    IMPACT_FIELDS,
    MESSAGE_IMPACT_SCHEMA,
    ImpactParseError,
    parse_message_impact,
)


def valid_payload() -> dict[str, object]:
    return {
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


def test_parse_strict_message_impact_json() -> None:
    impact = parse_message_impact(json.dumps(valid_payload()))
    assert impact.topic_relevance == 78
    assert impact.primary_topic == "philosophy_ideas"
    assert impact.disagreement_detected is True


def test_schema_is_strict_and_requires_every_field() -> None:
    assert set(MESSAGE_IMPACT_SCHEMA["required"]) == set(IMPACT_FIELDS)
    assert MESSAGE_IMPACT_SCHEMA["additionalProperties"] is False
    assert '"sigamos con lo anterior"' in EVALUATOR_SYSTEM_PROMPT
    assert "delega la eleccion" in EVALUATOR_SYSTEM_PROMPT


def test_parser_rejects_llm_suggested_deltas() -> None:
    payload = valid_payload()
    payload["suggested_deltas"] = {"candor": 4}  # type: ignore[assignment]
    with pytest.raises(ImpactParseError, match="sobran: suggested_deltas"):
        parse_message_impact(payload)


@pytest.mark.parametrize("bad_value", [-1, 101, 20.5, True, "50"])
def test_parser_rejects_invalid_scores(bad_value: object) -> None:
    payload = valid_payload()
    payload["novelty"] = bad_value  # type: ignore[assignment]
    with pytest.raises(ImpactParseError):
        parse_message_impact(payload)


def test_parser_rejects_markdown_or_non_object_json() -> None:
    with pytest.raises(ImpactParseError):
        parse_message_impact("```json\n{}\n```")
    with pytest.raises(ImpactParseError):
        parse_message_impact("[]")


def test_parser_accepts_legacy_only_when_explicitly_requested() -> None:
    payload = valid_payload()
    del payload["primary_topic"]
    del payload["secondary_topic"]
    with pytest.raises(ImpactParseError, match="faltan"):
        parse_message_impact(payload)
    impact = parse_message_impact(payload, allow_legacy=True)
    assert impact.primary_topic == "none"

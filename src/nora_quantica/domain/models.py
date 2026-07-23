from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

TOPIC_CATEGORIES = (
    "everyday_life",
    "personal_social",
    "arts_literature",
    "philosophy_ideas",
    "technology_science",
    "politics_society",
    "work_study",
    "practical_hobbies",
    "travel_nature_weather",
)
NO_TOPIC = "none"


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    """Limit a numeric value to an inclusive interval."""
    return min(maximum, max(minimum, value))


def _validate_score(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} debe ser numerico")
    if not 0 <= value <= 100:
        raise ValueError(f"{name} debe estar entre 0 y 100")


def _validate_integer_score(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} debe ser un entero")
    _validate_score(name, value)


@dataclass(frozen=True, slots=True)
class BaselineState:
    relational_closeness: int
    availability: int
    interest_bias: int
    candor: int
    topic_orientation: int
    topic_affinities: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = asdict(self)
        affinities = values.pop("topic_affinities")
        for name, value in values.items():
            _validate_integer_score(name, value)
        if set(affinities) != set(TOPIC_CATEGORIES):
            raise ValueError("Las afinidades tematicas deben incluir todas las categorias")
        for name, value in affinities.items():
            _validate_integer_score(name, value)


@dataclass(frozen=True, slots=True)
class CurrentState:
    """Dynamic values retain decimals internally to avoid recovery dead zones."""

    availability: float
    current_interest: float
    candor: float
    topic_orientation: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _validate_score(name, value)

    def display_values(self) -> dict[str, int]:
        return {name: round(value) for name, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class EntropyMetadata:
    provider: str
    retrieved_at: datetime
    raw_bytes_hash: str
    fallback_used: bool = False
    fallback_type: str | None = None
    response_id: str | None = None


@dataclass(frozen=True, slots=True)
class MessageImpact:
    """A descriptive evaluator output. It deliberately contains no state deltas."""

    topic_relevance: int
    novelty: int
    continuity: int
    disagreement_strength: int
    social_signal: int
    urgency: int
    event_intensity: int
    engagement_request: int
    topic_shift: int
    primary_topic: str
    secondary_topic: str

    def __post_init__(self) -> None:
        values = asdict(self)
        primary_topic = values.pop("primary_topic")
        secondary_topic = values.pop("secondary_topic")
        for name, value in values.items():
            _validate_integer_score(name, value)
        allowed = {*TOPIC_CATEGORIES, NO_TOPIC}
        if primary_topic not in allowed or secondary_topic not in allowed:
            raise ValueError("Categoria tematica desconocida")
        if primary_topic == NO_TOPIC and secondary_topic != NO_TOPIC:
            raise ValueError("No puede haber tema secundario sin tema primario")
        if primary_topic == secondary_topic and primary_topic != NO_TOPIC:
            raise ValueError("Los temas primario y secundario deben ser diferentes")

    @property
    def disagreement_detected(self) -> bool:
        return self.disagreement_strength > 0


@dataclass(frozen=True, slots=True)
class AppliedDeltas:
    requested: dict[str, float]
    effective: dict[str, float]
    max_delta: int


@dataclass(frozen=True, slots=True)
class StateTransition:
    previous: CurrentState
    current: CurrentState
    impact: MessageImpact
    deltas: AppliedDeltas


@dataclass(slots=True)
class ConversationState:
    conversation_id: str
    quantum_source: EntropyMetadata
    baseline: BaselineState
    current: CurrentState
    turn_number: int = 0
    last_transition: StateTransition | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

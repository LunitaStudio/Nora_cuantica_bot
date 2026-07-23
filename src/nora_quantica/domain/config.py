from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class InterestWeights:
    interest_bias: float = 0.40
    topic_affinity: float = 0.45
    novelty: float = 0.15
    topic_relevance: float = 0.0
    continuity: float = 0.0
    relational_closeness: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.interest_bias,
            self.topic_affinity,
            self.topic_relevance,
            self.novelty,
            self.continuity,
            self.relational_closeness,
        )
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("Los pesos de interes deben sumar 1.0")
        if any(value < 0 for value in values):
            raise ValueError("Los pesos de interes no pueden ser negativos")


@dataclass(frozen=True, slots=True)
class StateEngineConfig:
    normal_max_delta: int = 8
    high_impact_max_delta: int = 25
    high_impact_threshold: int = 75
    availability_recovery_rate: float = 0.05
    interest_weights: InterestWeights = field(default_factory=InterestWeights)
    preserve_interest_without_topic: bool = True


def interest_weights_from_dict(raw: dict) -> InterestWeights:
    if "topic_affinity" in raw:
        return InterestWeights(**raw)
    # Configuracion histórica: permite reconstruir sin alterar sesiones anteriores.
    return InterestWeights(
        interest_bias=raw.get("interest_bias", 0.30),
        topic_affinity=0.0,
        novelty=raw.get("novelty", 0.15),
        topic_relevance=raw.get("topic_relevance", 0.30),
        continuity=raw.get("continuity", 0.15),
        relational_closeness=raw.get("relational_closeness", 0.10),
    )


def state_engine_config_from_dict(raw: dict) -> StateEngineConfig:
    return StateEngineConfig(
        normal_max_delta=raw["normal_max_delta"],
        high_impact_max_delta=raw["high_impact_max_delta"],
        high_impact_threshold=raw["high_impact_threshold"],
        availability_recovery_rate=raw["availability_recovery_rate"],
        interest_weights=interest_weights_from_dict(raw.get("interest_weights", {})),
        preserve_interest_without_topic=raw.get("preserve_interest_without_topic", False),
    )

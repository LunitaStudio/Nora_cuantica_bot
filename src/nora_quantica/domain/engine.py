from __future__ import annotations

from nora_quantica.domain.config import StateEngineConfig
from nora_quantica.domain.models import (
    AppliedDeltas,
    ConversationState,
    CurrentState,
    MessageImpact,
    StateTransition,
    clamp,
)


class StateEngine:
    """Pure deterministic state transition engine."""

    def __init__(self, config: StateEngineConfig | None = None) -> None:
        self.config = config or StateEngineConfig()

    @staticmethod
    def calculate_topic_affinity(
        state: ConversationState,
        impact: MessageImpact,
    ) -> float:
        if impact.primary_topic == "none":
            return float(state.baseline.interest_bias)
        primary = state.baseline.topic_affinities[impact.primary_topic]
        if impact.secondary_topic == "none":
            return float(primary)
        secondary = state.baseline.topic_affinities[impact.secondary_topic]
        return 0.75 * primary + 0.25 * secondary

    def calculate_interest(
        self,
        state: ConversationState,
        impact: MessageImpact,
    ) -> float:
        weights = self.config.interest_weights
        if (
            impact.primary_topic == "none"
            and weights.topic_affinity > 0
            and self.config.preserve_interest_without_topic
        ):
            return state.current.current_interest
        topic_affinity = self.calculate_topic_affinity(state, impact)
        result = (
            weights.interest_bias * state.baseline.interest_bias
            + weights.topic_affinity * topic_affinity
            + weights.topic_relevance * impact.topic_relevance
            + weights.novelty * impact.novelty
            + weights.continuity * impact.continuity
            + weights.relational_closeness * state.baseline.relational_closeness
        )
        return clamp(result)

    def _requested_deltas(
        self,
        state: ConversationState,
        impact: MessageImpact,
    ) -> dict[str, float]:
        # Availability models bandwidth. Recovery is continuous, while demanding or
        # intense turns consume a small amount. Urgency changes response priority,
        # not the bot's fictional amount of free time.
        recovery = (
            state.baseline.availability - state.current.availability
        ) * self.config.availability_recovery_rate
        load = 2.0 * (impact.engagement_request / 100) + 1.0 * (impact.event_intensity / 100)
        availability = recovery - load

        # Disagreement invites clearer positioning. Strong affiliative signals
        # temper it slightly, without affecting factual honesty.
        candor = 6.0 * (impact.disagreement_strength / 100) - 1.5 * (
            impact.social_signal / 100
        )

        # Continuity and relevance anchor the thread; explicit topic shifts and
        # social exchange loosen it.
        topic_orientation = (
            3.0 * ((impact.continuity - 50) / 50)
            + 2.0 * ((impact.topic_relevance - 50) / 50)
            - 4.0 * (impact.topic_shift / 100)
            - 1.0 * (impact.social_signal / 100)
        )

        return {
            "availability": availability,
            "candor": candor,
            "topic_orientation": topic_orientation,
        }

    def apply(self, state: ConversationState, impact: MessageImpact) -> StateTransition:
        max_delta = (
            self.config.high_impact_max_delta
            if impact.event_intensity >= self.config.high_impact_threshold
            else self.config.normal_max_delta
        )
        requested = self._requested_deltas(state, impact)
        effective = {
            name: clamp(value, -max_delta, max_delta) for name, value in requested.items()
        }
        previous = state.current
        current = CurrentState(
            availability=clamp(previous.availability + effective["availability"]),
            current_interest=self.calculate_interest(state, impact),
            candor=clamp(previous.candor + effective["candor"]),
            topic_orientation=clamp(
                previous.topic_orientation + effective["topic_orientation"]
            ),
        )
        transition = StateTransition(
            previous=previous,
            current=current,
            impact=impact,
            deltas=AppliedDeltas(
                requested=requested,
                effective=effective,
                max_delta=max_delta,
            ),
        )
        state.current = current
        state.turn_number += 1
        state.last_transition = transition
        return transition

from dataclasses import replace

from nora_quantica.domain.behavior import (
    derive_controls,
    select_behavioral_plan,
    translate_behavioral_plan,
)
from nora_quantica.domain.models import CurrentState
from nora_quantica.prompt_profiles.baseline import BASELINE_PROFILE

from .test_engine import make_impact, make_state


def test_controls_reduce_rich_state_to_four_distinct_levers() -> None:
    state = make_state()
    state.current = CurrentState(
        availability=10,
        current_interest=90,
        candor=70,
        topic_orientation=80,
    )

    controls = derive_controls(state)

    assert set(controls.bands()) == {
        "social_distance",
        "response_budget",
        "topic_pull",
        "conversational_autonomy",
    }
    assert controls.response_budget < controls.topic_pull
    assert controls.social_distance == 28


def test_behavioral_move_is_deterministic_for_conversation_and_turn() -> None:
    state = make_state()
    state.turn_number = 3
    impact = make_impact()

    first = select_behavioral_plan(state, impact)
    second = select_behavioral_plan(state, impact)

    assert first == second
    assert first.roll_uint64 >= 0
    assert first.move in first.weights


def test_different_turns_derive_different_quantum_seeded_rolls() -> None:
    state = make_state()
    impact = make_impact()
    state.turn_number = 1
    first = select_behavioral_plan(state, impact)
    state.turn_number = 2
    second = select_behavioral_plan(state, impact)

    assert first.roll_uint64 != second.roll_uint64


def test_experiment_seed_reuses_roll_across_conversation_ids() -> None:
    first_state = make_state()
    second_state = make_state()
    first_state.conversation_id = "comparison-a"
    second_state.conversation_id = "comparison-b"
    first_state.metadata["behavior_seed_id"] = "generator-battery-v1"
    second_state.metadata["behavior_seed_id"] = "generator-battery-v1"
    first_state.turn_number = second_state.turn_number = 2

    first = select_behavioral_plan(first_state, make_impact())
    second = select_behavioral_plan(second_state, make_impact())

    assert first.roll_uint64 == second.roll_uint64
    assert first.move == second.move


def test_moves_are_only_offered_when_context_allows_them() -> None:
    state = make_state()
    neutral = make_impact(disagreement_strength=0)
    plan = select_behavioral_plan(state, neutral)
    assert "express_disagreement" not in plan.weights

    disagreement = make_impact(disagreement_strength=80)
    plan = select_behavioral_plan(state, disagreement)
    assert "express_disagreement" in plan.weights


def test_translation_contains_one_explicit_move_and_no_internal_state_values() -> None:
    state = make_state()
    plan = select_behavioral_plan(state, make_impact())

    instruction = translate_behavioral_plan(plan, BASELINE_PROFILE.safety_floor)

    assert "Jugada de este turno" in instruction
    assert BASELINE_PROFILE.safety_floor in instruction
    assert str(round(plan.controls.response_budget)) not in instruction


def test_only_question_move_authorizes_a_question() -> None:
    state = make_state()
    impact = make_impact()
    plan = select_behavioral_plan(state, impact)
    non_question = replace(plan, move="answer_and_association")
    question = replace(plan, move="answer_and_question")

    assert "no hagas preguntas" in translate_behavioral_plan(
        non_question,
        BASELINE_PROFILE.safety_floor,
    )
    assert "máximo una" in translate_behavioral_plan(
        question,
        BASELINE_PROFILE.safety_floor,
    )


def test_redirect_may_ask_only_about_the_new_topic() -> None:
    state = make_state()
    plan = replace(
        select_behavioral_plan(state, make_impact()),
        move="gentle_redirect",
    )

    instruction = translate_behavioral_plan(plan, BASELINE_PROFILE.safety_floor)

    assert "pregunta sobre el tema nuevo" in instruction
    assert "exclusivamente" in instruction


def test_low_budget_increases_answer_only_weight() -> None:
    high = make_state()
    low = make_state()
    low.current = replace(low.current, availability=0, current_interest=10)
    high.current = replace(high.current, availability=100, current_interest=90)
    impact = make_impact(engagement_request=20, disagreement_strength=0)

    low_plan = select_behavioral_plan(low, impact)
    high_plan = select_behavioral_plan(high, impact)

    assert low_plan.weights["answer_only"] > high_plan.weights["answer_only"]
    assert (
        low_plan.weights["answer_and_question"]
        < high_plan.weights["answer_and_question"]
    )

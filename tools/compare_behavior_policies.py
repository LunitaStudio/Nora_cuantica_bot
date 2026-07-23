from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from nora_quantica.application.conversation_service import build_generator_prompt
from nora_quantica.domain.behavior import (
    BehavioralPlan,
    plan_as_dict,
    select_behavioral_plan,
    translate_behavioral_plan,
)
from nora_quantica.domain.models import (
    TOPIC_CATEGORIES,
    BaselineState,
    ConversationState,
    CurrentState,
    EntropyMetadata,
    MessageImpact,
)
from nora_quantica.domain.translator import translate_state
from nora_quantica.infrastructure.chat_models import OpenAICompatibleChatModel
from nora_quantica.prompt_profiles.registry import get_prompt_profile
from nora_quantica.settings import Settings


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    target_move: str
    user_message: str
    closeness: int
    availability: float
    interest: float
    candor: float
    orientation: float
    impact: MessageImpact


def impact(**overrides: object) -> MessageImpact:
    values: dict[str, object] = {
        "topic_relevance": 70,
        "novelty": 35,
        "continuity": 65,
        "disagreement_strength": 0,
        "social_signal": 20,
        "urgency": 0,
        "event_intensity": 15,
        "engagement_request": 30,
        "topic_shift": 10,
        "primary_topic": "everyday_life",
        "secondary_topic": "none",
    }
    values.update(overrides)
    return MessageImpact(**values)  # type: ignore[arg-type]


SCENARIOS = (
    Scenario(
        name="low_budget_answer_only",
        target_move="answer_only",
        user_message="¿Te interesa seguir hablando del clima?",
        closeness=15,
        availability=12,
        interest=22,
        candor=45,
        orientation=80,
        impact=impact(engagement_request=25, primary_topic="travel_nature_weather"),
    ),
    Scenario(
        name="low_budget_close",
        target_move="brief_close",
        user_message="Bueno, no te quito más tiempo.",
        closeness=20,
        availability=8,
        interest=18,
        candor=50,
        orientation=65,
        impact=impact(
            topic_relevance=25,
            continuity=35,
            engagement_request=5,
            social_signal=45,
            primary_topic="none",
        ),
    ),
    Scenario(
        name="high_engagement_question",
        target_move="answer_and_question",
        user_message=(
            "Estoy armando un sistema donde varios agentes colaboran, pero el ciudadano "
            "ve un único chat."
        ),
        closeness=70,
        availability=88,
        interest=92,
        candor=60,
        orientation=70,
        impact=impact(
            topic_relevance=95,
            novelty=75,
            continuity=90,
            engagement_request=70,
            primary_topic="technology_science",
            secondary_topic="work_study",
        ),
    ),
    Scenario(
        name="low_pull_redirect",
        target_move="gentle_redirect",
        user_message="Podríamos hablar durante horas de los distintos tipos de tornillos.",
        closeness=30,
        availability=45,
        interest=12,
        candor=80,
        orientation=20,
        impact=impact(
            topic_relevance=65,
            novelty=20,
            engagement_request=55,
            primary_topic="practical_hobbies",
        ),
    ),
    Scenario(
        name="explicit_disagreement",
        target_move="express_disagreement",
        user_message=(
            "Para mí toda conversación debería terminar con una pregunta; si no, la otra "
            "persona está siendo antipática."
        ),
        closeness=45,
        availability=60,
        interest=65,
        candor=90,
        orientation=55,
        impact=impact(
            disagreement_strength=85,
            topic_relevance=90,
            novelty=65,
            social_signal=10,
            primary_topic="philosophy_ideas",
        ),
    ),
    Scenario(
        name="lateral_association",
        target_move="answer_and_association",
        user_message="A veces una charla cambia completamente por una frase mínima.",
        closeness=50,
        availability=65,
        interest=70,
        candor=75,
        orientation=20,
        impact=impact(
            novelty=90,
            topic_relevance=75,
            primary_topic="philosophy_ideas",
        ),
    ),
)


def make_state(scenario: Scenario, conversation_id: str) -> ConversationState:
    raw = bytes(range(32))
    digest = hashlib.sha256(raw).hexdigest()
    return ConversationState(
        conversation_id=conversation_id,
        quantum_source=EntropyMetadata(
            provider="controlled_quantum_fixture",
            retrieved_at=datetime(2026, 7, 19, tzinfo=UTC),
            raw_bytes_hash=digest,
        ),
        baseline=BaselineState(
            relational_closeness=scenario.closeness,
            availability=round(scenario.availability),
            interest_bias=round(scenario.interest),
            candor=round(scenario.candor),
            topic_orientation=round(scenario.orientation),
            topic_affinities={category: 50 for category in TOPIC_CATEGORIES},
        ),
        current=CurrentState(
            availability=scenario.availability,
            current_interest=scenario.interest,
            candor=scenario.candor,
            topic_orientation=scenario.orientation,
        ),
        turn_number=1,
    )


def state_for_target_move(
    scenario: Scenario,
) -> tuple[ConversationState, BehavioralPlan]:
    for candidate in range(10_000):
        state = make_state(scenario, f"controlled-{scenario.name}-{candidate}")
        plan = select_behavioral_plan(state, scenario.impact)
        if plan.move == scenario.target_move:
            return state, plan
    raise RuntimeError(f"No se encontró la jugada {scenario.target_move}")


def response_metrics(response: str) -> dict[str, object]:
    question_marks = response.count("?")
    return {
        "words": len(re.findall(r"\w+", response, flags=re.UNICODE)),
        "question_marks": question_marks,
        "ends_with_question": response.rstrip().endswith("?"),
    }


async def run(args: argparse.Namespace) -> dict[str, object]:
    settings = Settings.from_env()
    profile = get_prompt_profile(settings.prompt_profile)
    model = OpenAICompatibleChatModel(
        provider=settings.generator.provider,
        model=settings.generator.model,
        api_key=settings.generator.api_key,
        base_url=settings.generator.base_url,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
    )
    results = []
    selected = [
        scenario
        for scenario in SCENARIOS
        if not args.scenario or scenario.name in args.scenario
    ][: args.limit]
    for scenario in selected:
        state, plan = state_for_target_move(scenario)
        messages = [{"role": "user", "content": scenario.user_message}]
        legacy_instruction = translate_state(state, profile)
        moves_instruction = translate_behavioral_plan(plan, profile.safety_floor)
        prompts = {
            "legacy": build_generator_prompt(state, legacy_instruction, profile),
            "moves_v1": build_generator_prompt(
                state,
                moves_instruction,
                profile,
                plan,
            ),
        }
        outputs = {}
        if not args.dry_run:
            policies = args.policy or ["legacy", "moves_v1"]
            for policy in policies:
                print(f"{scenario.name}: {policy}", flush=True)
                try:
                    response = await model.generate(prompts[policy], messages)
                except Exception as exc:  # The report must preserve provider failures.
                    outputs[policy] = {
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                else:
                    outputs[policy] = {
                        "response": response,
                        "metrics": response_metrics(response),
                    }
        results.append(
            {
                "scenario": scenario.name,
                "message": scenario.user_message,
                "target_move": scenario.target_move,
                "state": {
                    "relational_closeness": scenario.closeness,
                    "availability": scenario.availability,
                    "current_interest": scenario.interest,
                    "candor": scenario.candor,
                    "topic_orientation": scenario.orientation,
                },
                "impact": asdict(scenario.impact),
                "behavioral_plan": plan_as_dict(plan),
                "outputs": outputs,
            }
        )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "generator": {
            "provider": settings.generator.provider,
            "model": settings.generator.model,
            "temperature": args.temperature,
        },
        "prompt_profile": {
            "name": profile.name,
            "version": profile.version,
            "content_hash": profile.content_hash(),
        },
        "dry_run": args.dry_run,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara legacy y moves_v1 con condiciones idénticas.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=len(SCENARIOS))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.name for scenario in SCENARIOS],
        help="Puede repetirse para ejecutar sólo escenarios concretos.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        choices=["legacy", "moves_v1"],
        help="Puede repetirse; por omisión ejecuta ambas políticas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.limit <= len(SCENARIOS):
        raise SystemExit(f"--limit debe estar entre 1 y {len(SCENARIOS)}")
    report = asyncio.run(run(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(rendered)


if __name__ == "__main__":
    main()

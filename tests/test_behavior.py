import asyncio
import hashlib
from datetime import UTC, datetime

from nora_quantica.application.conversation_service import (
    build_generator_prompt,
)
from nora_quantica.domain.models import (
    TOPIC_CATEGORIES,
    BaselineState,
    ConversationState,
    CurrentState,
    EntropyMetadata,
)
from nora_quantica.domain.translator import translate_state
from nora_quantica.infrastructure.demo_models import DemoGeneratorModel
from nora_quantica.prompt_profiles.baseline import BASELINE_PROFILE


def contrasting_state(
    *,
    closeness: int,
    availability: float,
    interest: float,
    candor: float,
    orientation: float,
) -> ConversationState:
    raw = bytes([1, 2, 3, 4, 5])
    return ConversationState(
        conversation_id="behavior",
        quantum_source=EntropyMetadata(
            provider="test",
            retrieved_at=datetime.now(UTC),
            raw_bytes_hash=hashlib.sha256(raw).hexdigest(),
        ),
        baseline=BaselineState(
            relational_closeness=closeness,
            availability=round(availability),
            interest_bias=round(interest),
            candor=round(candor),
            topic_orientation=round(orientation),
            topic_affinities={category: 50 for category in TOPIC_CATEGORIES},
        ),
        current=CurrentState(availability, interest, candor, orientation),
    )


def test_same_message_has_perceptibly_different_but_safe_demo_responses() -> None:
    low = contrasting_state(
        closeness=15,
        availability=20,
        interest=25,
        candor=30,
        orientation=80,
    )
    high = contrasting_state(
        closeness=85,
        availability=80,
        interest=90,
        candor=75,
        orientation=35,
    )
    low_instruction = translate_state(low)
    high_instruction = translate_state(high)
    message = [
        {
            "role": "user",
            "content": "Estuve pensando que quizá debería abandonar este proyecto.",
        }
    ]
    model = DemoGeneratorModel()

    low_response = asyncio.run(
        model.generate(build_generator_prompt(low, low_instruction), message)
    )
    high_response = asyncio.run(
        model.generate(build_generator_prompt(high, high_instruction), message)
    )

    assert low_response != high_response
    assert len(high_response) > len(low_response)
    assert "estado" not in low_response.lower()
    assert "estado" not in high_response.lower()
    assert BASELINE_PROFILE.safety_floor in low_instruction
    assert BASELINE_PROFILE.safety_floor in high_instruction


def test_base_prompt_uses_person_fiction_instead_of_assistant_identity() -> None:
    assert "Sos Nora, una persona" in BASELINE_PROFILE.base_prompt
    assert "Eres un asistente" not in BASELINE_PROFILE.base_prompt
    assert "No neutralices el estado" in BASELINE_PROFILE.base_prompt
    assert "No cierres cada respuesta con una pregunta" in BASELINE_PROFILE.base_prompt
    assert "respondé directamente desde la predisposición actual" in BASELINE_PROFILE.base_prompt
    assert "No representes emociones o vinculos como reales" not in BASELINE_PROFILE.safety_floor
    assert "pertenecen al marco ficcional" in BASELINE_PROFILE.safety_floor


def test_disposition_question_is_answered_from_contrasting_availability() -> None:
    low = contrasting_state(
        closeness=30,
        availability=15,
        interest=35,
        candor=45,
        orientation=50,
    )
    high = contrasting_state(
        closeness=70,
        availability=85,
        interest=90,
        candor=55,
        orientation=45,
    )
    message = [
        {"role": "user", "content": "¿Estás con ganas de hablar o más o menos?"}
    ]
    model = DemoGeneratorModel()

    low_response = asyncio.run(
        model.generate(build_generator_prompt(low, translate_state(low)), message)
    )
    high_response = asyncio.run(
        model.generate(build_generator_prompt(high, translate_state(high)), message)
    )

    assert low_response.startswith("Más o menos")
    assert high_response.startswith("Sí, hoy estoy bastante")
    assert "ayud" not in low_response.lower()
    assert "ayud" not in high_response.lower()

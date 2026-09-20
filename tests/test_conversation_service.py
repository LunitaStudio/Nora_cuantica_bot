import asyncio
import hashlib
import json
from datetime import UTC, datetime

import pytest

from nora_quantica.application.conversation_service import ConversationService
from nora_quantica.domain.entropy import EntropySample
from nora_quantica.domain.models import EntropyMetadata
from nora_quantica.infrastructure.sqlite_store import SQLiteStore
from nora_quantica.prompt_profiles import PromptProfile
from nora_quantica.prompt_profiles.baseline import BASELINE_PROFILE

from .test_evaluator import valid_payload


class Entropy:
    async def acquire(self):
        raw = bytes(range(32))
        return EntropySample(
            raw,
            EntropyMetadata(
                provider="test_quantum",
                retrieved_at=datetime.now(UTC),
                raw_bytes_hash=hashlib.sha256(raw).hexdigest(),
            ),
        )


class Model:
    def __init__(self, output: str | Exception, model: str):
        self.output = output
        self._model = model
        self.calls = []

    @property
    def provider(self):
        return "test"

    @property
    def model(self):
        return self._model

    async def generate(self, system_prompt, messages, response_schema=None):
        self.calls.append((system_prompt, messages, response_schema))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def make_service(
    tmp_path,
    generator_output="Respuesta natural",
    prompt_profile: PromptProfile = BASELINE_PROFILE,
    behavior_policy: str = "moves_v1",
):
    store = SQLiteStore(tmp_path / "nora.db")
    store.initialize()
    evaluator = Model(json.dumps(valid_payload()), "evaluator")
    generator = Model(generator_output, "generator")
    return (
        ConversationService(
            store=store,
            entropy=Entropy(),
            evaluator=evaluator,
            generator=generator,
            prompt_profile=prompt_profile,
            behavior_policy=behavior_policy,
        ),
        store,
        evaluator,
        generator,
    )


def test_complete_turn_is_generated_and_auditable(tmp_path) -> None:
    service, store, evaluator, generator = make_service(tmp_path)

    state = asyncio.run(service.create_conversation())
    result = asyncio.run(service.send_message(state.conversation_id, "Hola, ¿seguimos?"))

    assert result.response == "Respuesta natural"
    assert result.turn_number == 1
    assert len(store.list_messages(state.conversation_id)) == 2
    assert evaluator.calls[0][2] is not None
    assert generator.calls[0][2] is None
    assert "Controles conductuales estructurados" in generator.calls[0][0]
    assert '"relational_closeness"' not in generator.calls[0][0]
    assert '"candor"' not in generator.calls[0][0]
    assert '"conversational_move"' in generator.calls[0][0]
    assert "Traduccion conductual" in generator.calls[0][0]
    assert "Personalidad temática" in generator.calls[0][0]
    assert "Afinidades más" in generator.calls[0][0]

    lab = service.laboratory_snapshot(state.conversation_id)
    assert lab["last_turn"]["impact"]["topic_relevance"] == 78
    assert lab["last_turn"]["generator_model"] == "generator"
    assert lab["prompt_profile"]["name"] == "baseline"
    assert lab["last_topic_affinity"] is not None
    assert lab["behavior_policy"] == "moves_v1"
    assert lab["last_turn"]["behavioral_plan"]["move"]
    assert len(lab["baseline"]["topic_affinities"]) == 9
    exported = service.export_session(state.conversation_id)
    assert len(exported["turns"]) == 1
    assert exported["metadata"]["prompt_profile"]["content_hash"]
    assert exported["metadata"]["behavior_policy"]["name"] == "moves_v1"


def test_generator_failure_does_not_persist_or_advance_turn(tmp_path) -> None:
    service, store, _, _ = make_service(tmp_path, RuntimeError("provider down"))
    state = asyncio.run(service.create_conversation())

    with pytest.raises(RuntimeError, match="provider down"):
        asyncio.run(service.send_message(state.conversation_id, "Hola"))

    restored = store.get_conversation(state.conversation_id)
    assert restored is not None
    assert restored.turn_number == 0
    assert store.list_messages(state.conversation_id) == []


def test_history_is_forwarded_on_second_turn(tmp_path) -> None:
    service, _, evaluator, _ = make_service(tmp_path)
    state = asyncio.run(service.create_conversation())
    asyncio.run(service.send_message(state.conversation_id, "Primer mensaje"))
    asyncio.run(service.send_message(state.conversation_id, "Segundo mensaje"))

    second_history = evaluator.calls[1][1]
    assert [message["role"] for message in second_history] == [
        "user",
        "assistant",
        "user",
    ]
    assert second_history[-1]["content"] == "Segundo mensaje"


def test_conversation_stops_before_calling_models_at_turn_limit(tmp_path) -> None:
    service, store, evaluator, generator = make_service(tmp_path)
    service.max_turns = 1
    state = asyncio.run(service.create_conversation())
    asyncio.run(service.send_message(state.conversation_id, "Primer mensaje"))

    with pytest.raises(RuntimeError, match="límite de 1 turnos"):
        asyncio.run(service.send_message(state.conversation_id, "Segundo mensaje"))

    restored = store.get_conversation(state.conversation_id)
    assert restored is not None and restored.turn_number == 1
    assert len(evaluator.calls) == 1
    assert len(generator.calls) == 1


def test_conversation_keeps_profile_snapshot_when_active_profile_changes(tmp_path) -> None:
    first = PromptProfile(
        name="experimental",
        version="test-a",
        base_prompt="PROMPT FIJO A",
        instructions=BASELINE_PROFILE.instructions,
        safety_floor=BASELINE_PROFILE.safety_floor,
    )
    second = PromptProfile(
        name="experimental",
        version="test-b",
        base_prompt="PROMPT NUEVO B",
        instructions=BASELINE_PROFILE.instructions,
        safety_floor=BASELINE_PROFILE.safety_floor,
    )
    service, _, _, generator = make_service(tmp_path, prompt_profile=first)
    state = asyncio.run(service.create_conversation())
    service.prompt_profile = second

    asyncio.run(service.send_message(state.conversation_id, "Hola"))

    generated_prompt = generator.calls[0][0]
    assert "PROMPT FIJO A" in generated_prompt
    assert "PROMPT NUEVO B" not in generated_prompt
    lab = service.laboratory_snapshot(state.conversation_id)
    assert lab["prompt_profile"]["version"] == "test-a"


def test_legacy_policy_remains_selectable_for_controlled_comparisons(tmp_path) -> None:
    service, _, _, generator = make_service(tmp_path, behavior_policy="legacy")
    state = asyncio.run(service.create_conversation())

    asyncio.run(service.send_message(state.conversation_id, "Hola"))

    prompt = generator.calls[0][0]
    assert "Estado conversacional estructurado" in prompt
    assert "Controles conductuales estructurados" not in prompt
    exported = service.export_session(state.conversation_id)
    assert exported["turns"][0]["behavioral_plan"] is None

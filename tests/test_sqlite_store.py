import hashlib
from datetime import UTC, datetime

from nora_quantica.application.ports import TurnRecord
from nora_quantica.domain.engine import StateEngine
from nora_quantica.domain.entropy import EntropySample, initial_state_from_entropy
from nora_quantica.domain.models import EntropyMetadata
from nora_quantica.domain.translator import translate_state
from nora_quantica.infrastructure.sqlite_store import SQLiteStore

from .test_engine import make_impact


def entropy(raw: bytes, response_id: str = "test-response") -> EntropySample:
    return EntropySample(
        raw_bytes=raw,
        metadata=EntropyMetadata(
            provider="test_qrng",
            retrieved_at=datetime.now(UTC),
            raw_bytes_hash=hashlib.sha256(raw).hexdigest(),
            response_id=response_id,
        ),
    )


def test_store_reconstructs_and_exports_a_complete_turn(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "nora.db")
    store.initialize()
    sample = entropy(bytes(range(32)))
    state = initial_state_from_entropy(sample, conversation_id="conversation-1")
    store.create_conversation(state, sample.raw_bytes)

    impact = make_impact()
    transition = StateEngine().apply(state, impact)
    instruction = translate_state(state)
    store.save_turn(
        state,
        TurnRecord(
            conversation_id=state.conversation_id,
            turn_number=state.turn_number,
            user_message="Hola",
            assistant_message="Hola, ¿cómo estás?",
            impact=impact,
            transition=transition,
            behavioral_instruction=instruction,
            evaluator_prompt="eval prompt",
            evaluator_output="{}",
            generator_prompt="generator prompt",
            evaluator_provider="test",
            evaluator_model="eval",
            generator_provider="test",
            generator_model="gen",
            evaluator_latency_ms=12,
            generator_latency_ms=34,
            created_at=datetime.now(UTC),
        ),
    )

    restored = store.get_conversation("conversation-1")
    assert restored is not None
    assert restored.turn_number == 1
    assert restored.current == state.current
    assert restored.baseline == state.baseline
    assert [message.role for message in store.list_messages("conversation-1")] == [
        "user",
        "assistant",
    ]

    exported = store.export_session("conversation-1")
    assert exported is not None
    assert exported["quantum_source"]["raw_bytes"] == list(range(32))
    assert exported["turns"][0]["impact"]["novelty"] == impact.novelty
    assert exported["turns"][0]["behavioral_instruction"] == instruction


def test_entropy_cache_consumes_without_reusing_bytes(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "nora.db")
    store.initialize()
    store.put(entropy(bytes(range(10)), "batch-1"))

    first = store.take(6)
    second = store.take(4)
    empty = store.take(1)

    assert first is not None and first.raw_bytes == bytes(range(6))
    assert second is not None and second.raw_bytes == bytes(range(6, 10))
    assert first.metadata.fallback_type == "local_quantum_cache"
    assert empty is None

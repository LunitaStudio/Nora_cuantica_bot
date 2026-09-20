from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

from nora_quantica.application.ports import ChatMessage, TurnRecord
from nora_quantica.domain.entropy import EntropySample, topic_affinities_from_bytes
from nora_quantica.domain.models import (
    BaselineState,
    ConversationState,
    CurrentState,
    EntropyMetadata,
)
from nora_quantica.infrastructure.sqlite_store import _as_jsonable


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _plan_for_firestore(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    result = dict(plan)
    roll = result.get("roll_uint64")
    if isinstance(roll, int) and roll > 2**63 - 1:
        result["roll_uint64"] = str(roll)
    return result


def _plan_from_firestore(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    result = dict(plan)
    roll = result.get("roll_uint64")
    if isinstance(roll, str) and roll.isdigit():
        result["roll_uint64"] = int(roll)
    return result


class FirestoreStore:
    """Firestore implementation of conversation persistence and entropy caching."""

    def __init__(
        self,
        *,
        project_id: str | None = None,
        collection_prefix: str = "nora",
        retention_hours: int = 48,
        client: Any | None = None,
    ) -> None:
        if retention_hours <= 0:
            raise ValueError("La retención de Firestore debe ser positiva")
        if not collection_prefix.strip():
            raise ValueError("El prefijo de colecciones no puede estar vacío")
        self.project_id = project_id
        self.collection_prefix = collection_prefix.strip()
        self.retention = timedelta(hours=retention_hours)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                from google.cloud import firestore
            except ImportError as exc:
                raise RuntimeError(
                    "Firestore requiere instalar la dependencia opcional 'cloud'"
                ) from exc
            self._client = firestore.Client(project=self.project_id)
        return self._client

    @property
    def conversations(self):
        return self.client.collection(f"{self.collection_prefix}_conversations")

    @property
    def entropy_cache(self):
        return self.client.collection(f"{self.collection_prefix}_entropy_cache")

    def initialize(self) -> None:
        # Firestore crea colecciones y documentos al escribir por primera vez.
        _ = self.client

    def _expiry(self, value: datetime | None = None) -> datetime:
        return _utc(value or datetime.now(UTC)) + self.retention

    def create_conversation(self, state: ConversationState, raw_bytes: bytes) -> None:
        now = datetime.now(UTC)
        self.conversations.document(state.conversation_id).create(
            {
                "conversation_id": state.conversation_id,
                "created_at": now,
                "updated_at": now,
                "expires_at": self._expiry(now),
                "quantum_source": _as_jsonable(state.quantum_source),
                "raw_bytes": raw_bytes,
                "baseline": _as_jsonable(state.baseline),
                "current": _as_jsonable(state.current),
                "turn_number": state.turn_number,
                "metadata": state.metadata,
            }
        )

    def get_conversation(self, conversation_id: str) -> ConversationState | None:
        snapshot = self.conversations.document(conversation_id).get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict()
        raw_bytes = bytes(data["raw_bytes"])
        baseline_data = dict(data["baseline"])
        baseline_data.setdefault(
            "topic_affinities",
            topic_affinities_from_bytes(raw_bytes),
        )
        source = data["quantum_source"]
        return ConversationState(
            conversation_id=conversation_id,
            quantum_source=EntropyMetadata(
                provider=source["provider"],
                retrieved_at=datetime.fromisoformat(source["retrieved_at"]),
                raw_bytes_hash=source["raw_bytes_hash"],
                fallback_used=bool(source.get("fallback_used", False)),
                fallback_type=source.get("fallback_type"),
                response_id=source.get("response_id"),
            ),
            baseline=BaselineState(**baseline_data),
            current=CurrentState(**data["current"]),
            turn_number=int(data["turn_number"]),
            metadata=dict(data["metadata"]),
        )

    def list_messages(self, conversation_id: str) -> list[ChatMessage]:
        documents = (
            self.conversations.document(conversation_id)
            .collection("messages")
            .order_by("sequence")
            .stream()
        )
        result = []
        for document in documents:
            data = document.to_dict()
            result.append(
                ChatMessage(
                    role=data["role"],
                    content=data["content"],
                    created_at=_utc(data["created_at"]),
                    turn_number=int(data["turn_number"]),
                )
            )
        return result

    def save_turn(self, state: ConversationState, record: TurnRecord) -> None:
        if state.last_transition is None:
            raise ValueError("El estado no contiene una transición para persistir")
        if record.turn_number != state.turn_number:
            raise ValueError("El número de turno no coincide con el estado")

        conversation = self.conversations.document(state.conversation_id)
        transition = record.transition
        turn_id = f"{state.turn_number:06d}"
        expires_at = self._expiry(record.created_at)
        batch = self.client.batch()
        batch.create(
            conversation.collection("messages").document(f"{turn_id}-user"),
            {
                "sequence": state.turn_number * 2,
                "role": "user",
                "content": record.user_message,
                "turn_number": state.turn_number,
                "created_at": record.created_at,
                "expires_at": expires_at,
            },
        )
        batch.create(
            conversation.collection("messages").document(f"{turn_id}-assistant"),
            {
                "sequence": state.turn_number * 2 + 1,
                "role": "assistant",
                "content": record.assistant_message,
                "turn_number": state.turn_number,
                "created_at": record.created_at,
                "expires_at": expires_at,
            },
        )
        batch.create(
            conversation.collection("turns").document(turn_id),
            {
                "conversation_id": state.conversation_id,
                "turn_number": state.turn_number,
                "impact": _as_jsonable(record.impact),
                "previous_state": _as_jsonable(transition.previous),
                "current_state": _as_jsonable(transition.current),
                "requested_deltas": transition.deltas.requested,
                "effective_deltas": transition.deltas.effective,
                "max_delta": transition.deltas.max_delta,
                "behavioral_instruction": record.behavioral_instruction,
                "behavioral_plan": _plan_for_firestore(record.behavioral_plan),
                "evaluator_prompt": record.evaluator_prompt,
                "evaluator_output": record.evaluator_output,
                "generator_prompt": record.generator_prompt,
                "evaluator_provider": record.evaluator_provider,
                "evaluator_model": record.evaluator_model,
                "generator_provider": record.generator_provider,
                "generator_model": record.generator_model,
                "evaluator_latency_ms": record.evaluator_latency_ms,
                "generator_latency_ms": record.generator_latency_ms,
                "created_at": record.created_at,
                "expires_at": expires_at,
            },
        )
        batch.update(
            conversation,
            {
                "updated_at": record.created_at,
                "expires_at": expires_at,
                "current": _as_jsonable(state.current),
                "turn_number": state.turn_number,
                "metadata": state.metadata,
            },
        )
        batch.commit()

    def latest_turn(self, conversation_id: str) -> dict[str, Any] | None:
        documents = (
            self.conversations.document(conversation_id)
            .collection("turns")
            .order_by("turn_number", direction="DESCENDING")
            .limit(1)
            .stream()
        )
        for document in documents:
            return self._turn_data(document.to_dict())
        return None

    def _turn_data(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        result.pop("expires_at", None)
        result["behavioral_plan"] = _plan_from_firestore(
            result.get("behavioral_plan")
        )
        if isinstance(result.get("created_at"), datetime):
            result["created_at"] = _utc(result["created_at"]).isoformat()
        return _as_jsonable(result)

    def export_session(self, conversation_id: str) -> dict[str, Any] | None:
        snapshot = self.conversations.document(conversation_id).get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict()
        state = self.get_conversation(conversation_id)
        assert state is not None
        turns = (
            self.conversations.document(conversation_id)
            .collection("turns")
            .order_by("turn_number")
            .stream()
        )
        source = dict(data["quantum_source"])
        source["raw_bytes"] = list(bytes(data["raw_bytes"]))
        return {
            "conversation_id": conversation_id,
            "created_at": _utc(data["created_at"]).isoformat(),
            "updated_at": _utc(data["updated_at"]).isoformat(),
            "quantum_source": source,
            "baseline": _as_jsonable(state.baseline),
            "current": _as_jsonable(state.current),
            "turn_number": state.turn_number,
            "metadata": state.metadata,
            "messages": [_as_jsonable(item) for item in self.list_messages(conversation_id)],
            "turns": [self._turn_data(document.to_dict()) for document in turns],
        }

    def put(self, sample: EntropySample) -> None:
        now = datetime.now(UTC)
        self.entropy_cache.document().create(
            {
                "provider": sample.metadata.provider,
                "retrieved_at": sample.metadata.retrieved_at,
                "raw_bytes_hash": sample.metadata.raw_bytes_hash,
                "raw_bytes": sample.raw_bytes,
                "response_id": sample.metadata.response_id,
                "created_at": now,
                "expires_at": self._expiry(now),
            }
        )

    def take(self, count: int) -> EntropySample | None:
        if count <= 0:
            raise ValueError("La cantidad debe ser positiva")
        snapshots = list(self.entropy_cache.order_by("created_at").stream())
        selected = []
        total = 0
        for snapshot in snapshots:
            selected.append(snapshot)
            total += len(bytes(snapshot.to_dict()["raw_bytes"]))
            if total >= count:
                break
        if total < count:
            return None

        combined = b"".join(bytes(item.to_dict()["raw_bytes"]) for item in selected)
        consumed, remainder = combined[:count], combined[count:]
        batch = self.client.batch()
        for snapshot in selected:
            batch.delete(snapshot.reference)
        if remainder:
            last = selected[-1].to_dict()
            remainder_ref = self.entropy_cache.document()
            batch.create(
                remainder_ref,
                {
                    "provider": last["provider"],
                    "retrieved_at": last["retrieved_at"],
                    "raw_bytes_hash": hashlib.sha256(remainder).hexdigest(),
                    "raw_bytes": remainder,
                    "response_id": last.get("response_id"),
                    "created_at": datetime.now(UTC),
                    "expires_at": self._expiry(),
                },
            )
        batch.commit()
        return EntropySample(
            raw_bytes=consumed,
            metadata=EntropyMetadata(
                provider="quantum_cache",
                retrieved_at=datetime.now(UTC),
                raw_bytes_hash=hashlib.sha256(consumed).hexdigest(),
                fallback_used=True,
                fallback_type="firestore_quantum_cache",
                response_id="+".join(
                    str(item.to_dict().get("response_id") or item.id) for item in selected
                ),
            ),
        )

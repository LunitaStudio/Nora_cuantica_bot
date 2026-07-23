from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nora_quantica.application.ports import ChatMessage, TurnRecord
from nora_quantica.domain.entropy import EntropySample, topic_affinities_from_bytes
from nora_quantica.domain.evaluator import parse_message_impact
from nora_quantica.domain.models import (
    AppliedDeltas,
    BaselineState,
    ConversationState,
    CurrentState,
    EntropyMetadata,
    StateTransition,
)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: _as_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    return value


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    raw_bytes_hash TEXT NOT NULL,
    raw_bytes BLOB NOT NULL,
    fallback_used INTEGER NOT NULL,
    fallback_type TEXT,
    response_id TEXT,
    baseline_json TEXT NOT NULL,
    current_json TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    impact_json TEXT NOT NULL,
    previous_state_json TEXT NOT NULL,
    current_state_json TEXT NOT NULL,
    requested_deltas_json TEXT NOT NULL,
    effective_deltas_json TEXT NOT NULL,
    max_delta INTEGER NOT NULL,
    behavioral_instruction TEXT NOT NULL,
    behavioral_plan_json TEXT,
    evaluator_prompt TEXT NOT NULL,
    evaluator_output TEXT NOT NULL,
    generator_prompt TEXT NOT NULL,
    evaluator_provider TEXT NOT NULL,
    evaluator_model TEXT NOT NULL,
    generator_provider TEXT NOT NULL,
    generator_model TEXT NOT NULL,
    evaluator_latency_ms INTEGER NOT NULL,
    generator_latency_ms INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(conversation_id, turn_number)
);

CREATE TABLE IF NOT EXISTS entropy_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    raw_bytes_hash TEXT NOT NULL,
    raw_bytes BLOB NOT NULL,
    response_id TEXT,
    created_at TEXT NOT NULL
);
"""


class SQLiteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(turns)").fetchall()
            }
            if "behavioral_plan_json" not in columns:
                connection.execute(
                    "ALTER TABLE turns ADD COLUMN behavioral_plan_json TEXT"
                )

    def create_conversation(self, state: ConversationState, raw_bytes: bytes) -> None:
        now = datetime.now(UTC)
        source = state.quantum_source
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    conversation_id, created_at, updated_at, provider, retrieved_at,
                    raw_bytes_hash, raw_bytes, fallback_used, fallback_type, response_id,
                    baseline_json, current_json, turn_number, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.conversation_id,
                    _iso(now),
                    _iso(now),
                    source.provider,
                    _iso(source.retrieved_at),
                    source.raw_bytes_hash,
                    raw_bytes,
                    int(source.fallback_used),
                    source.fallback_type,
                    source.response_id,
                    _json(_as_jsonable(state.baseline)),
                    _json(_as_jsonable(state.current)),
                    state.turn_number,
                    _json(state.metadata),
                ),
            )

    def get_conversation(self, conversation_id: str) -> ConversationState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        baseline_data = json.loads(row["baseline_json"])
        baseline_data.setdefault(
            "topic_affinities",
            topic_affinities_from_bytes(bytes(row["raw_bytes"])),
        )
        baseline = BaselineState(**baseline_data)
        current = CurrentState(**json.loads(row["current_json"]))
        return ConversationState(
            conversation_id=row["conversation_id"],
            quantum_source=EntropyMetadata(
                provider=row["provider"],
                retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
                raw_bytes_hash=row["raw_bytes_hash"],
                fallback_used=bool(row["fallback_used"]),
                fallback_type=row["fallback_type"],
                response_id=row["response_id"],
            ),
            baseline=baseline,
            current=current,
            turn_number=row["turn_number"],
            metadata=json.loads(row["metadata_json"]),
        )

    def list_messages(self, conversation_id: str) -> list[ChatMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, created_at, turn_number
                FROM messages WHERE conversation_id = ? ORDER BY id
                """,
                (conversation_id,),
            ).fetchall()
        return [
            ChatMessage(
                role=row["role"],
                content=row["content"],
                created_at=datetime.fromisoformat(row["created_at"]),
                turn_number=row["turn_number"],
            )
            for row in rows
        ]

    def save_turn(self, state: ConversationState, record: TurnRecord) -> None:
        if state.last_transition is None:
            raise ValueError("El estado no contiene una transicion para persistir")
        if record.turn_number != state.turn_number:
            raise ValueError("El numero de turno no coincide con el estado")
        transition = record.transition
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (conversation_id, turn_number, role, content, created_at)
                VALUES (?, ?, 'user', ?, ?), (?, ?, 'assistant', ?, ?)
                """,
                (
                    state.conversation_id,
                    state.turn_number,
                    record.user_message,
                    _iso(record.created_at),
                    state.conversation_id,
                    state.turn_number,
                    record.assistant_message,
                    _iso(record.created_at),
                ),
            )
            connection.execute(
                """
                INSERT INTO turns (
                    conversation_id, turn_number, impact_json, previous_state_json,
                    current_state_json, requested_deltas_json, effective_deltas_json,
                    max_delta, behavioral_instruction, behavioral_plan_json,
                    evaluator_prompt, evaluator_output,
                    generator_prompt, evaluator_provider, evaluator_model,
                    generator_provider, generator_model, evaluator_latency_ms,
                    generator_latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.conversation_id,
                    state.turn_number,
                    _json(_as_jsonable(record.impact)),
                    _json(_as_jsonable(transition.previous)),
                    _json(_as_jsonable(transition.current)),
                    _json(transition.deltas.requested),
                    _json(transition.deltas.effective),
                    transition.deltas.max_delta,
                    record.behavioral_instruction,
                    (
                        _json(record.behavioral_plan)
                        if record.behavioral_plan is not None
                        else None
                    ),
                    record.evaluator_prompt,
                    record.evaluator_output,
                    record.generator_prompt,
                    record.evaluator_provider,
                    record.evaluator_model,
                    record.generator_provider,
                    record.generator_model,
                    record.evaluator_latency_ms,
                    record.generator_latency_ms,
                    _iso(record.created_at),
                ),
            )
            cursor = connection.execute(
                """
                UPDATE conversations SET updated_at = ?, current_json = ?,
                    turn_number = ?, metadata_json = ? WHERE conversation_id = ?
                """,
                (
                    _iso(record.created_at),
                    _json(_as_jsonable(state.current)),
                    state.turn_number,
                    _json(state.metadata),
                    state.conversation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Conversacion inexistente: {state.conversation_id}")

    def latest_turn(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM turns WHERE conversation_id = ? ORDER BY turn_number DESC LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        return self._turn_row(row) if row is not None else None

    def _turn_row(self, row: sqlite3.Row) -> dict[str, Any]:
        json_fields = {
            "impact_json": "impact",
            "previous_state_json": "previous_state",
            "current_state_json": "current_state",
            "requested_deltas_json": "requested_deltas",
            "effective_deltas_json": "effective_deltas",
        }
        result = dict(row)
        for source, target in json_fields.items():
            result[target] = json.loads(result.pop(source))
        raw_plan = result.pop("behavioral_plan_json", None)
        result["behavioral_plan"] = json.loads(raw_plan) if raw_plan else None
        result.pop("id", None)
        return result

    def export_session(self, conversation_id: str) -> dict[str, Any] | None:
        state = self.get_conversation(conversation_id)
        if state is None:
            return None
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            turns = connection.execute(
                "SELECT * FROM turns WHERE conversation_id = ? ORDER BY turn_number",
                (conversation_id,),
            ).fetchall()
        assert conversation is not None
        return {
            "conversation_id": conversation_id,
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "quantum_source": {
                "provider": conversation["provider"],
                "retrieved_at": conversation["retrieved_at"],
                "raw_bytes_hash": conversation["raw_bytes_hash"],
                "raw_bytes": list(conversation["raw_bytes"]),
                "fallback_used": bool(conversation["fallback_used"]),
                "fallback_type": conversation["fallback_type"],
                "response_id": conversation["response_id"],
            },
            "baseline": _as_jsonable(state.baseline),
            "current": _as_jsonable(state.current),
            "turn_number": state.turn_number,
            "metadata": state.metadata,
            "messages": [_as_jsonable(message) for message in self.list_messages(conversation_id)],
            "turns": [self._turn_row(row) for row in turns],
        }

    def put(self, sample: EntropySample) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO entropy_cache (
                    provider, retrieved_at, raw_bytes_hash, raw_bytes, response_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sample.metadata.provider,
                    _iso(sample.metadata.retrieved_at),
                    sample.metadata.raw_bytes_hash,
                    sample.raw_bytes,
                    sample.metadata.response_id,
                    _iso(datetime.now(UTC)),
                ),
            )

    def take(self, count: int) -> EntropySample | None:
        if count <= 0:
            raise ValueError("La cantidad debe ser positiva")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM entropy_cache ORDER BY id"
            ).fetchall()
            selected: list[sqlite3.Row] = []
            total = 0
            for row in rows:
                selected.append(row)
                total += len(row["raw_bytes"])
                if total >= count:
                    break
            if total < count:
                return None
            combined = b"".join(row["raw_bytes"] for row in selected)
            consumed, remainder = combined[:count], combined[count:]
            ids = [row["id"] for row in selected]
            placeholders = ",".join("?" for _ in ids)
            connection.execute(f"DELETE FROM entropy_cache WHERE id IN ({placeholders})", ids)
            if remainder:
                last = selected[-1]
                connection.execute(
                    """
                    INSERT INTO entropy_cache (
                        provider, retrieved_at, raw_bytes_hash, raw_bytes, response_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        last["provider"],
                        last["retrieved_at"],
                        hashlib.sha256(remainder).hexdigest(),
                        remainder,
                        last["response_id"],
                        _iso(datetime.now(UTC)),
                    ),
                )
        return EntropySample(
            raw_bytes=consumed,
            metadata=EntropyMetadata(
                provider="quantum_cache",
                retrieved_at=datetime.now(UTC),
                raw_bytes_hash=hashlib.sha256(consumed).hexdigest(),
                fallback_used=True,
                fallback_type="local_quantum_cache",
                response_id="+".join(str(row["response_id"] or row["id"]) for row in selected),
            ),
        )


def transition_from_dict(data: dict[str, Any]) -> StateTransition:
    """Public reconstruction helper used by audits and tests."""
    return StateTransition(
        previous=CurrentState(**data["previous_state"]),
        current=CurrentState(**data["current_state"]),
        impact=parse_message_impact(data["impact"], allow_legacy=True),
        deltas=AppliedDeltas(
            requested=data["requested_deltas"],
            effective=data["effective_deltas"],
            max_delta=data["max_delta"],
        ),
    )

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from nora_quantica.domain.models import (
    TOPIC_CATEGORIES,
    BaselineState,
    ConversationState,
    CurrentState,
    EntropyMetadata,
)

DISPOSITION_BYTES = 5
INITIAL_STATE_BYTES = DISPOSITION_BYTES + len(TOPIC_CATEGORIES)


def byte_to_score(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("El byte debe ser un entero")
    if not 0 <= value <= 255:
        raise ValueError("Byte fuera de rango")
    return round((value / 255) * 100)


@dataclass(frozen=True, slots=True)
class EntropySample:
    raw_bytes: bytes
    metadata: EntropyMetadata

    def __post_init__(self) -> None:
        if not self.raw_bytes:
            raise ValueError("La muestra de entropia no puede estar vacia")
        expected_hash = hashlib.sha256(self.raw_bytes).hexdigest()
        if self.metadata.raw_bytes_hash != expected_hash:
            raise ValueError("El hash no coincide con los bytes recibidos")


class EntropyProvider(Protocol):
    async def get_bytes(self, count: int) -> EntropySample: ...


class OsCsprngProvider:
    """Safe final fallback backed by the operating system CSPRNG."""

    async def get_bytes(self, count: int) -> EntropySample:
        if count <= 0:
            raise ValueError("La cantidad debe ser positiva")
        raw_bytes = secrets.token_bytes(count)
        return EntropySample(
            raw_bytes=raw_bytes,
            metadata=EntropyMetadata(
                provider="os_csprng",
                retrieved_at=datetime.now(UTC),
                raw_bytes_hash=hashlib.sha256(raw_bytes).hexdigest(),
                fallback_used=True,
                fallback_type="os_csprng",
            ),
        )


def topic_affinities_from_bytes(raw_bytes: bytes) -> dict[str, int]:
    if len(raw_bytes) < INITIAL_STATE_BYTES:
        raise ValueError(f"Se requieren al menos {INITIAL_STATE_BYTES} bytes")
    return {
        category: byte_to_score(raw_bytes[DISPOSITION_BYTES + index])
        for index, category in enumerate(TOPIC_CATEGORIES)
    }


def initial_state_from_entropy(
    sample: EntropySample,
    *,
    conversation_id: str | None = None,
) -> ConversationState:
    if len(sample.raw_bytes) < INITIAL_STATE_BYTES:
        raise ValueError(f"Se requieren al menos {INITIAL_STATE_BYTES} bytes")
    scores = [byte_to_score(value) for value in sample.raw_bytes[:DISPOSITION_BYTES]]
    topic_affinities = topic_affinities_from_bytes(sample.raw_bytes)
    baseline = BaselineState(
        relational_closeness=scores[0],
        availability=scores[1],
        interest_bias=scores[2],
        candor=scores[3],
        topic_orientation=scores[4],
        topic_affinities=topic_affinities,
    )
    current = CurrentState(
        availability=float(baseline.availability),
        current_interest=float(baseline.interest_bias),
        candor=float(baseline.candor),
        topic_orientation=float(baseline.topic_orientation),
    )
    return ConversationState(
        conversation_id=conversation_id or str(uuid.uuid4()),
        quantum_source=sample.metadata,
        baseline=baseline,
        current=current,
    )

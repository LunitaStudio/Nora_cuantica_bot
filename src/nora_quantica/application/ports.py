from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from nora_quantica.domain.entropy import EntropySample
from nora_quantica.domain.models import ConversationState, MessageImpact, StateTransition


class ChatModel(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str
    created_at: datetime
    turn_number: int


@dataclass(frozen=True, slots=True)
class TurnRecord:
    conversation_id: str
    turn_number: int
    user_message: str
    assistant_message: str
    impact: MessageImpact
    transition: StateTransition
    behavioral_instruction: str
    evaluator_prompt: str
    evaluator_output: str
    generator_prompt: str
    evaluator_provider: str
    evaluator_model: str
    generator_provider: str
    generator_model: str
    evaluator_latency_ms: int
    generator_latency_ms: int
    created_at: datetime
    behavioral_plan: dict[str, Any] | None = None


class ConversationStore(Protocol):
    def initialize(self) -> None: ...

    def create_conversation(self, state: ConversationState, raw_bytes: bytes) -> None: ...

    def get_conversation(self, conversation_id: str) -> ConversationState | None: ...

    def list_messages(self, conversation_id: str) -> list[ChatMessage]: ...

    def save_turn(self, state: ConversationState, record: TurnRecord) -> None: ...

    def latest_turn(self, conversation_id: str) -> dict[str, Any] | None: ...

    def export_session(self, conversation_id: str) -> dict[str, Any] | None: ...


class EntropyCache(Protocol):
    def put(self, sample: EntropySample) -> None: ...

    def take(self, count: int) -> EntropySample | None: ...

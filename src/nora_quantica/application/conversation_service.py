from __future__ import annotations

import asyncio
import hmac
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Protocol

from nora_quantica.application.audit import audit_session
from nora_quantica.application.ports import (
    ChatMessage,
    ChatModel,
    ConversationStore,
    TurnRecord,
)
from nora_quantica.domain.behavior import (
    LEGACY_POLICY,
    MOVES_POLICY,
    SUPPORTED_POLICIES,
    BehavioralPlan,
    plan_as_dict,
    policy_snapshot,
    select_behavioral_plan,
    translate_behavioral_plan,
)
from nora_quantica.domain.config import state_engine_config_from_dict
from nora_quantica.domain.engine import StateEngine
from nora_quantica.domain.entropy import EntropySample, initial_state_from_entropy
from nora_quantica.domain.evaluator import (
    EVALUATOR_SYSTEM_PROMPT,
    MESSAGE_IMPACT_SCHEMA,
    parse_message_impact,
)
from nora_quantica.domain.models import ConversationState, MessageImpact, StateTransition
from nora_quantica.domain.translator import translate_state, translate_topic_personality
from nora_quantica.prompt_profiles import PromptProfile
from nora_quantica.prompt_profiles.baseline import BASELINE_PROFILE
from nora_quantica.prompt_profiles.registry import profile_from_snapshot

MAX_MESSAGE_LENGTH = 20_000

class EntropyAcquirer(Protocol):
    async def acquire(self) -> EntropySample: ...


class ConversationNotFoundError(KeyError):
    pass


class ConversationTurnLimitError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TurnResult:
    conversation_id: str
    response: str
    turn_number: int
    current_state: dict[str, int]
    impact: MessageImpact
    transition: StateTransition
    behavioral_instruction: str
    behavioral_plan: dict[str, Any] | None


def public_state(state: ConversationState) -> dict[str, int]:
    return {
        "relational_closeness": state.baseline.relational_closeness,
        **state.current.display_values(),
    }


def build_generator_prompt(
    state: ConversationState,
    behavioral_instruction: str,
    profile: PromptProfile = BASELINE_PROFILE,
    behavioral_plan: BehavioralPlan | None = None,
) -> str:
    if behavioral_plan is None:
        prompt_state = public_state(state)
        state_label = "Estado conversacional estructurado"
    else:
        prompt_state = {
            "controls": asdict(behavioral_plan.controls),
            "control_bands": behavioral_plan.controls.bands(),
            "conversational_move": behavioral_plan.move,
        }
        state_label = "Controles conductuales estructurados"
    state_json = json.dumps(prompt_state, ensure_ascii=False, indent=2)
    topic_personality = translate_topic_personality(state)
    return (
        f"{profile.base_prompt}\n\n"
        f"{state_label}:\n{state_json}\n\n"
        f"Traduccion conductual:\n{behavioral_instruction}\n\n"
        f"Personalidad temática:\n{topic_personality}"
    )


class ConversationService:
    def __init__(
        self,
        *,
        store: ConversationStore,
        entropy: EntropyAcquirer,
        evaluator: ChatModel,
        generator: ChatModel,
        state_engine: StateEngine | None = None,
        history_limit: int = 24,
        max_turns: int = 24,
        prompt_profile: PromptProfile = BASELINE_PROFILE,
        behavior_policy: str = MOVES_POLICY,
        behavior_seed_id: str | None = None,
    ) -> None:
        if behavior_policy not in SUPPORTED_POLICIES:
            raise ValueError(f"Política conductual desconocida: {behavior_policy}")
        if history_limit <= 0:
            raise ValueError("El límite de historial debe ser positivo")
        if max_turns <= 0:
            raise ValueError("El límite de turnos debe ser positivo")
        self.store = store
        self.entropy = entropy
        self.evaluator = evaluator
        self.generator = generator
        self.state_engine = state_engine or StateEngine()
        self.history_limit = history_limit
        self.max_turns = max_turns
        self.prompt_profile = prompt_profile
        self.behavior_policy = behavior_policy
        self.behavior_seed_id = behavior_seed_id
        self._locks: dict[str, asyncio.Lock] = {}

    async def create_conversation(self, owner_session: str | None = None) -> ConversationState:
        sample = await self.entropy.acquire()
        state = initial_state_from_entropy(sample)
        state.metadata["state_engine_config"] = asdict(self.state_engine.config)
        state.metadata["prompt_profile"] = self.prompt_profile.snapshot()
        state.metadata["behavior_policy"] = policy_snapshot(self.behavior_policy)
        if owner_session is not None:
            state.metadata["owner_session"] = owner_session
        if self.behavior_seed_id is not None:
            state.metadata["behavior_seed_id"] = self.behavior_seed_id
        self.store.create_conversation(state, sample.raw_bytes)
        return state

    def assert_owner(self, conversation_id: str, owner_session: str) -> None:
        state = self.get_conversation(conversation_id)
        stored_owner = state.metadata.get("owner_session")
        if stored_owner is not None and not hmac.compare_digest(
            str(stored_owner),
            owner_session,
        ):
            raise ConversationNotFoundError(conversation_id)

    def get_conversation(self, conversation_id: str) -> ConversationState:
        state = self.store.get_conversation(conversation_id)
        if state is None:
            raise ConversationNotFoundError(conversation_id)
        return state

    def messages(self, conversation_id: str) -> list[ChatMessage]:
        self.get_conversation(conversation_id)
        return self.store.list_messages(conversation_id)

    async def send_message(self, conversation_id: str, message: str) -> TurnResult:
        clean_message = message.strip()
        if not clean_message:
            raise ValueError("El mensaje no puede estar vacio")
        if len(clean_message) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"El mensaje supera el limite de {MAX_MESSAGE_LENGTH} caracteres")

        lock = self._locks.setdefault(conversation_id, asyncio.Lock())
        async with lock:
            return await self._send_locked(conversation_id, clean_message)

    async def _send_locked(self, conversation_id: str, message: str) -> TurnResult:
        state = self.get_conversation(conversation_id)
        if state.turn_number >= self.max_turns:
            raise ConversationTurnLimitError(
                f"La conversación alcanzó el límite de {self.max_turns} turnos"
            )
        profile = self._profile_for_state(state)
        stored_history = self.store.list_messages(conversation_id)
        history = [
            {"role": item.role, "content": item.content}
            for item in stored_history[-self.history_limit :]
        ]
        messages_with_user = [*history, {"role": "user", "content": message}]

        evaluator_started = perf_counter()
        evaluator_output = await self.evaluator.generate(
            EVALUATOR_SYSTEM_PROMPT,
            messages_with_user,
            MESSAGE_IMPACT_SCHEMA,
        )
        evaluator_latency_ms = round((perf_counter() - evaluator_started) * 1000)
        impact = parse_message_impact(evaluator_output)

        transition = self._engine_for_state(state).apply(state, impact)
        behavior_policy = self._behavior_policy_for_state(state)
        behavioral_plan = None
        if behavior_policy == MOVES_POLICY:
            behavioral_plan = select_behavioral_plan(state, impact)
            instruction = translate_behavioral_plan(
                behavioral_plan,
                profile.safety_floor,
            )
        else:
            instruction = translate_state(state, profile)
        generator_prompt = build_generator_prompt(
            state,
            instruction,
            profile,
            behavioral_plan,
        )

        generator_started = perf_counter()
        response = await self.generator.generate(
            generator_prompt,
            messages_with_user,
        )
        generator_latency_ms = round((perf_counter() - generator_started) * 1000)
        response = response.strip()
        if not response:
            raise ValueError("El generador devolvio una respuesta vacia")

        created_at = datetime.now(UTC)
        record = TurnRecord(
            conversation_id=conversation_id,
            turn_number=state.turn_number,
            user_message=message,
            assistant_message=response,
            impact=impact,
            transition=transition,
            behavioral_instruction=instruction,
            behavioral_plan=(
                plan_as_dict(behavioral_plan) if behavioral_plan is not None else None
            ),
            evaluator_prompt=EVALUATOR_SYSTEM_PROMPT,
            evaluator_output=evaluator_output,
            generator_prompt=generator_prompt,
            evaluator_provider=self.evaluator.provider,
            evaluator_model=self.evaluator.model,
            generator_provider=self.generator.provider,
            generator_model=self.generator.model,
            evaluator_latency_ms=evaluator_latency_ms,
            generator_latency_ms=generator_latency_ms,
            created_at=created_at,
        )
        self.store.save_turn(state, record)
        return TurnResult(
            conversation_id=conversation_id,
            response=response,
            turn_number=state.turn_number,
            current_state=public_state(state),
            impact=impact,
            transition=transition,
            behavioral_instruction=instruction,
            behavioral_plan=(
                plan_as_dict(behavioral_plan) if behavioral_plan is not None else None
            ),
        )

    def laboratory_snapshot(self, conversation_id: str) -> dict[str, Any]:
        exported = self.store.export_session(conversation_id)
        if exported is None:
            raise ConversationNotFoundError(conversation_id)
        latest = self.store.latest_turn(conversation_id)
        state = self.get_conversation(conversation_id)
        topic_affinity = None
        if latest is not None:
            latest_impact = parse_message_impact(latest["impact"], allow_legacy=True)
            topic_affinity = self._engine_for_state(state).calculate_topic_affinity(
                state,
                latest_impact,
            )
        snapshot = exported.get("metadata", {}).get("prompt_profile")
        profile = (
            profile_from_snapshot(snapshot)
            if isinstance(snapshot, dict)
            else self.prompt_profile
        )
        return {
            "conversation_id": conversation_id,
            "quantum_source": exported["quantum_source"],
            "baseline": exported["baseline"],
            "current": exported["current"],
            "turn_number": exported["turn_number"],
            "prompt_profile": {
                "name": profile.name,
                "version": profile.version,
                "content_hash": profile.content_hash(),
            },
            "behavior_policy": self._behavior_policy_for_state(state),
            "last_turn": latest,
            "last_topic_affinity": topic_affinity,
            "audit": asdict(audit_session(exported)),
        }

    def _profile_for_state(self, state: ConversationState) -> PromptProfile:
        snapshot = state.metadata.get("prompt_profile")
        if not isinstance(snapshot, dict):
            # Las conversaciones anteriores a los perfiles quedan fijadas al
            # perfil activo la primera vez que vuelven a usarse.
            snapshot = self.prompt_profile.snapshot()
            state.metadata["prompt_profile"] = snapshot
        return profile_from_snapshot(snapshot)

    def _engine_for_state(self, state: ConversationState) -> StateEngine:
        raw = state.metadata.get("state_engine_config")
        if not isinstance(raw, dict):
            return self.state_engine
        return StateEngine(state_engine_config_from_dict(raw))

    def _behavior_policy_for_state(self, state: ConversationState) -> str:
        raw = state.metadata.get("behavior_policy")
        if not isinstance(raw, dict):
            return LEGACY_POLICY
        name = raw.get("name")
        return name if name in SUPPORTED_POLICIES else LEGACY_POLICY

    def export_session(self, conversation_id: str) -> dict[str, Any]:
        exported = self.store.export_session(conversation_id)
        if exported is None:
            raise ConversationNotFoundError(conversation_id)
        metadata = dict(exported.get("metadata", {}))
        metadata.pop("owner_session", None)
        exported["metadata"] = metadata
        return exported

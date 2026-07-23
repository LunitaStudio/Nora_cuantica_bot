from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from nora_quantica.domain.behavior import (
    MOVES_POLICY,
    SUPPORTED_POLICIES,
    plan_as_dict,
    select_behavioral_plan,
)
from nora_quantica.domain.config import state_engine_config_from_dict
from nora_quantica.domain.engine import StateEngine
from nora_quantica.domain.entropy import EntropySample, initial_state_from_entropy
from nora_quantica.domain.evaluator import parse_message_impact
from nora_quantica.domain.models import EntropyMetadata
from nora_quantica.prompt_profiles import PromptProfile


@dataclass(frozen=True, slots=True)
class AuditReport:
    passed: bool
    checks: int
    errors: list[str]


def _close_dict(actual: dict[str, Any], expected: dict[str, Any], tolerance: float = 1e-9) -> bool:
    if set(actual) != set(expected):
        return False
    for key in actual:
        left, right = actual[key], expected[key]
        if isinstance(left, int | float) and isinstance(right, int | float):
            if abs(left - right) > tolerance:
                return False
        elif left != right:
            return False
    return True


def _close_nested(actual: Any, expected: Any, tolerance: float = 1e-9) -> bool:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _close_nested(actual[key], expected[key], tolerance) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _close_nested(left, right, tolerance)
            for left, right in zip(actual, expected, strict=True)
        )
    if (
        isinstance(actual, int | float)
        and not isinstance(actual, bool)
        and isinstance(expected, int | float)
        and not isinstance(expected, bool)
    ):
        return abs(actual - expected) <= tolerance
    return actual == expected


def _engine_from_export(exported: dict[str, Any]) -> StateEngine:
    raw = exported.get("metadata", {}).get("state_engine_config")
    if not isinstance(raw, dict):
        return StateEngine()
    return StateEngine(state_engine_config_from_dict(raw))


def audit_session(exported: dict[str, Any]) -> AuditReport:
    errors: list[str] = []
    checks = 0
    behavior_snapshot = exported.get("metadata", {}).get("behavior_policy")
    behavior_policy = None
    if behavior_snapshot is not None:
        checks += 1
        if (
            not isinstance(behavior_snapshot, dict)
            or behavior_snapshot.get("name") not in SUPPORTED_POLICIES
        ):
            errors.append("Política conductual inválida")
        else:
            behavior_policy = behavior_snapshot["name"]
    profile_snapshot = exported.get("metadata", {}).get("prompt_profile")
    if profile_snapshot is not None:
        checks += 1
        try:
            PromptProfile.from_snapshot(profile_snapshot)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"Perfil de prompt inválido: {exc}")
    source = exported.get("quantum_source", {})
    try:
        raw_bytes = bytes(source["raw_bytes"])
        digest = hashlib.sha256(raw_bytes).hexdigest()
        checks += 1
        if digest != source.get("raw_bytes_hash"):
            errors.append("El hash SHA-256 no coincide con los bytes registrados")
        sample = EntropySample(
            raw_bytes,
            EntropyMetadata(
                provider=source.get("provider", "audit"),
                retrieved_at=datetime.now(UTC),
                raw_bytes_hash=digest,
                fallback_used=bool(source.get("fallback_used")),
                fallback_type=source.get("fallback_type"),
                response_id=source.get("response_id"),
            ),
        )
        replay = initial_state_from_entropy(
            sample,
            conversation_id=exported.get("conversation_id", "audit"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return AuditReport(False, checks, [f"Fuente de entropía inválida: {exc}"])

    checks += 1
    if exported.get("baseline") != asdict(replay.baseline):
        errors.append("El baseline no deriva de los bytes de estado registrados")

    try:
        engine = _engine_from_export(exported)
    except (KeyError, TypeError, ValueError) as exc:
        return AuditReport(False, checks, [*errors, f"Configuración del motor inválida: {exc}"])

    turns = exported.get("turns", [])
    if not isinstance(turns, list):
        return AuditReport(False, checks, [*errors, "La lista de turnos es inválida"])

    for expected_turn, turn in enumerate(turns, start=1):
        checks += 1
        if turn.get("turn_number") != expected_turn:
            errors.append(f"Turno {expected_turn}: numeración no consecutiva")
        previous = {
            "availability": replay.current.availability,
            "current_interest": replay.current.current_interest,
            "candor": replay.current.candor,
            "topic_orientation": replay.current.topic_orientation,
        }
        if not _close_dict(turn.get("previous_state", {}), previous):
            errors.append(f"Turno {expected_turn}: estado previo inconsistente")
        try:
            transition = engine.apply(
                replay,
                parse_message_impact(turn["impact"], allow_legacy=True),
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"Turno {expected_turn}: impacto inválido ({exc})")
            continue
        calculated_current = {
            "availability": transition.current.availability,
            "current_interest": transition.current.current_interest,
            "candor": transition.current.candor,
            "topic_orientation": transition.current.topic_orientation,
        }
        if not _close_dict(turn.get("current_state", {}), calculated_current):
            errors.append(f"Turno {expected_turn}: estado calculado inconsistente")
        if not _close_dict(turn.get("effective_deltas", {}), transition.deltas.effective):
            errors.append(f"Turno {expected_turn}: deltas efectivos inconsistentes")
        if turn.get("max_delta") != transition.deltas.max_delta:
            errors.append(f"Turno {expected_turn}: límite de delta inconsistente")
        if behavior_policy == MOVES_POLICY:
            checks += 1
            expected_plan = plan_as_dict(
                select_behavioral_plan(replay, transition.impact)
            )
            if not _close_nested(turn.get("behavioral_plan"), expected_plan):
                errors.append(
                    f"Turno {expected_turn}: plan conductual inconsistente"
                )

    checks += 1
    final_current = {
        "availability": replay.current.availability,
        "current_interest": replay.current.current_interest,
        "candor": replay.current.candor,
        "topic_orientation": replay.current.topic_orientation,
    }
    if not _close_dict(exported.get("current", {}), final_current):
        errors.append("El estado final no coincide con el replay")
    if exported.get("turn_number") != len(turns):
        errors.append("El contador final no coincide con los turnos registrados")
    messages = exported.get("messages", [])
    checks += 1
    if len(messages) != 2 * len(turns):
        errors.append("La cantidad de mensajes no coincide con los turnos")
    else:
        roles = [message.get("role") for message in messages]
        if roles != [role for _ in turns for role in ("user", "assistant")]:
            errors.append("El orden de roles de los mensajes es inconsistente")
    return AuditReport(not errors, checks, errors)

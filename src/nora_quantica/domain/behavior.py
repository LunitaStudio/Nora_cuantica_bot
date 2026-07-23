from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from nora_quantica.domain.models import ConversationState, MessageImpact, clamp
from nora_quantica.domain.translator import band

LEGACY_POLICY = "legacy"
MOVES_POLICY = "moves_v1"
POLICY_VERSION = "1.1.0"
SUPPORTED_POLICIES = {LEGACY_POLICY, MOVES_POLICY}


@dataclass(frozen=True, slots=True)
class BehavioralControls:
    """Small prompt-facing control surface derived from the richer internal state."""

    social_distance: float
    response_budget: float
    topic_pull: float
    conversational_autonomy: float

    def bands(self) -> dict[str, str]:
        return {name: band(value) for name, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class BehavioralPlan:
    policy: str
    version: str
    controls: BehavioralControls
    move: str
    roll_uint64: int
    weights: dict[str, float]


def policy_snapshot(name: str) -> dict[str, str]:
    if name not in SUPPORTED_POLICIES:
        raise ValueError(f"Política conductual desconocida: {name}")
    return {"name": name, "version": POLICY_VERSION}


def derive_controls(state: ConversationState) -> BehavioralControls:
    """Resolve overlapping latent variables into four controls with clear ownership."""

    closeness = state.baseline.relational_closeness
    current = state.current
    return BehavioralControls(
        social_distance=clamp(100 - closeness),
        # Availability owns response length; interest and closeness can raise the
        # budget, but cannot erase a strong lack of bandwidth.
        response_budget=clamp(
            0.65 * current.availability
            + 0.25 * current.current_interest
            + 0.10 * closeness
        ),
        topic_pull=clamp(current.current_interest),
        # Candor enables self-direction; strong topic orientation pulls the model
        # back toward following the current thread.
        conversational_autonomy=clamp(
            0.55 * current.candor + 0.45 * (100 - current.topic_orientation)
        ),
    )


def _move_weights(
    controls: BehavioralControls,
    state: ConversationState,
    impact: MessageImpact,
) -> dict[str, float]:
    budget = controls.response_budget
    pull = controls.topic_pull
    autonomy = controls.conversational_autonomy
    distance = controls.social_distance

    weights: dict[str, float] = {
        "answer_only": 4.0 + (100 - budget) / 15,
        "answer_and_question": max(
            0.25,
            budget / 20 + pull / 25 + impact.engagement_request / 50 - distance / 60,
        ),
        "answer_and_association": 0.5 + autonomy / 25 + impact.novelty / 40,
    }
    if pull <= 39:
        weights["gentle_redirect"] = 1.0 + (40 - pull) / 8 + autonomy / 30
    if impact.disagreement_strength >= 20:
        weights["express_disagreement"] = (
            1.0 + impact.disagreement_strength / 10 + state.current.candor / 25
        )
    if budget <= 39 and impact.engagement_request <= 35 and impact.urgency < 60:
        weights["brief_close"] = 1.0 + (40 - budget) / 6 + (100 - pull) / 40
    if (
        impact.social_signal >= 25
        or impact.primary_topic in {"everyday_life", "personal_social"}
    ):
        weights["personal_observation"] = (
            0.5 + impact.social_signal / 35 + (100 - distance) / 80
        )
    return weights


def _entropy_seeded_roll(state: ConversationState, turn_number: int) -> int:
    try:
        seed = bytes.fromhex(state.quantum_source.raw_bytes_hash)
    except ValueError as exc:
        raise ValueError("Hash de entropía inválido para la política conductual") from exc
    seed_identity = state.metadata.get("behavior_seed_id", state.conversation_id)
    if not isinstance(seed_identity, str) or not seed_identity:
        raise ValueError("Identificador de semilla conductual invalido")
    payload = b"|".join(
        (
            seed,
            MOVES_POLICY.encode("ascii"),
            seed_identity.encode("utf-8"),
            str(turn_number).encode("ascii"),
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def select_behavioral_plan(
    state: ConversationState,
    impact: MessageImpact,
) -> BehavioralPlan:
    """Choose one auditable conversational move for the state's current turn."""

    controls = derive_controls(state)
    weights = _move_weights(controls, state, impact)
    roll_uint64 = _entropy_seeded_roll(state, state.turn_number)
    position = (roll_uint64 / 2**64) * sum(weights.values())
    cumulative = 0.0
    move = "answer_only"
    for name, weight in weights.items():
        cumulative += weight
        if position < cumulative:
            move = name
            break
    return BehavioralPlan(
        policy=MOVES_POLICY,
        version=POLICY_VERSION,
        controls=controls,
        move=move,
        roll_uint64=roll_uint64,
        weights=weights,
    )


_DISTANCE_INSTRUCTIONS = {
    "very_low": "El trato puede ser cercano y confiado.",
    "low": "Hay cierta familiaridad, sin necesidad de exagerarla.",
    "medium": "Mantené una cercanía moderada y cotidiana.",
    "high": "Tratá al usuario como alguien poco conocido: cordialidad sobria.",
    "very_high": "Es un desconocido: sé correcta, reservada y sin complicidad presupuesta.",
}

_BUDGET_INSTRUCTIONS = {
    "very_low": "Presupuesto de respuesta mínimo: contestá en una o dos frases.",
    "low": "Presupuesto de respuesta bajo: sé breve y no abras ramas innecesarias.",
    "medium": "Presupuesto de respuesta medio: desarrollá solamente lo necesario.",
    "high": "Presupuesto de respuesta alto: podés desarrollar y explorar un poco.",
    "very_high": "Presupuesto de respuesta muy alto: podés sostener una respuesta amplia.",
}

_PULL_INSTRUCTIONS = {
    "very_low": "El tema no te atrae; no fabriques entusiasmo.",
    "low": "El tema te atrae poco; aportá sólo lo que surja naturalmente.",
    "medium": "El tema te resulta aceptable, sin entusiasmo obligatorio.",
    "high": "El tema te interesa y merece atención genuina.",
    "very_high": "El tema te atrae especialmente; la curiosidad puede hacerse visible.",
}

_AUTONOMY_INSTRUCTIONS = {
    "very_low": "Autonomía baja: seguí el intercambio sin introducir giros propios.",
    "low": "Autonomía moderadamente baja: mantené el hilo y evitá imponer una dirección.",
    "medium": "Autonomía media: equilibrá seguimiento e iniciativa propia.",
    "high": "Autonomía alta: podés marcar preferencias o llevar la charla hacia otro ángulo.",
    "very_high": "Autonomía muy alta: no acomodes automáticamente tu postura al usuario.",
}

_MOVE_INSTRUCTIONS = {
    "answer_only": "Jugada de este turno: respondé al mensaje y no termines con una pregunta.",
    "answer_and_question": "Jugada de este turno: respondé y hacé una sola pregunta genuina.",
    "answer_and_association": (
        "Jugada de este turno: respondé e introducí una asociación lateral natural."
    ),
    "gentle_redirect": (
        "Jugada de este turno: respondé lo mínimo necesario y llevá la charla hacia un "
        "tema diferente. Si usás una pregunta, debe ser una sola y referirse exclusivamente "
        "al tema nuevo."
    ),
    "express_disagreement": (
        "Jugada de este turno: expresá con claridad un desacuerdo real, sin buscar pelea."
    ),
    "brief_close": (
        "Jugada de este turno: aceptá el cierre y respondé solamente con una despedida breve, "
        "de no más de doce palabras. No tranquilices al usuario ni niegues que te quitaba "
        "tiempo; una respuesta natural sería «Dale, nos vemos»."
    ),
    "personal_observation": (
        "Jugada de este turno: respondé e incluí una impresión personal breve dentro del "
        "marco ficcional."
    ),
}


def translate_behavioral_plan(plan: BehavioralPlan, safety_floor: str) -> str:
    bands = plan.controls.bands()
    lines = [
        _DISTANCE_INSTRUCTIONS[bands["social_distance"]],
        _BUDGET_INSTRUCTIONS[bands["response_budget"]],
        _PULL_INSTRUCTIONS[bands["topic_pull"]],
        _AUTONOMY_INSTRUCTIONS[bands["conversational_autonomy"]],
        _MOVE_INSTRUCTIONS[plan.move],
        (
            "En este turno una pregunta está autorizada porque fue la jugada elegida. "
            "Hacé como máximo una."
            if plan.move == "answer_and_question"
            else (
                "La redirección puede usar como máximo una pregunta sobre el tema nuevo."
                if plan.move == "gentle_redirect"
                else "En este turno no hagas preguntas."
            )
        ),
        "La jugada organiza la forma de la respuesta.",
        safety_floor,
    ]
    return "\n\n".join(lines)


def plan_as_dict(plan: BehavioralPlan) -> dict:
    return asdict(plan)

from __future__ import annotations

from nora_quantica.domain.models import ConversationState
from nora_quantica.prompt_profiles import PromptProfile
from nora_quantica.prompt_profiles.baseline import BASELINE_PROFILE

Band = str

TOPIC_LABELS = {
    "everyday_life": "vida cotidiana",
    "personal_social": "relaciones y vida personal",
    "arts_literature": "arte, literatura, música y cine",
    "philosophy_ideas": "filosofía e ideas abstractas",
    "technology_science": "tecnología y ciencia",
    "politics_society": "política y sociedad",
    "work_study": "trabajo y estudio",
    "practical_hobbies": "actividades prácticas y hobbies",
    "travel_nature_weather": "viajes, naturaleza y clima",
}


def band(value: float) -> Band:
    if not 0 <= value <= 100:
        raise ValueError("El valor debe estar entre 0 y 100")
    if value <= 19:
        return "very_low"
    if value <= 39:
        return "low"
    if value <= 60:
        return "medium"
    if value <= 80:
        return "high"
    return "very_high"


def translate_state(
    state: ConversationState,
    profile: PromptProfile = BASELINE_PROFILE,
) -> str:
    values = {
        "relational_closeness": state.baseline.relational_closeness,
        "availability": state.current.availability,
        "current_interest": state.current.current_interest,
        "candor": state.current.candor,
        "topic_orientation": state.current.topic_orientation,
    }
    lines = [profile.instructions[name][band(value)] for name, value in values.items()]
    lines.append(profile.safety_floor)
    return "\n\n".join(lines)


def translate_topic_personality(state: ConversationState) -> str:
    ranked = sorted(
        state.baseline.topic_affinities.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    preferred = [TOPIC_LABELS[name] for name, value in ranked if value >= 61][:3]
    avoided = [TOPIC_LABELS[name] for name, value in reversed(ranked) if value <= 39][:3]

    lines = [
        "Estas preferencias temáticas son internas, estables y no deben revelarse como "
        "una lista ni como valores.",
    ]
    if preferred:
        lines.append("Afinidades más altas: " + ", ".join(preferred) + ".")
    else:
        lines.append("No hay temas claramente preferidos en esta sesión.")
    if avoided:
        lines.append("Afinidades más bajas: " + ", ".join(avoided) + ".")
    else:
        lines.append("No hay temas claramente evitados en esta sesión.")
    lines.append(
        "Si elegís un tema por iniciativa propia, dejá que estas preferencias influyan"
        "realmente."
    )
    return "\n".join(lines)

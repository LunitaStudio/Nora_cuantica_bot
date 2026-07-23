from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from nora_quantica.domain.models import NO_TOPIC, TOPIC_CATEGORIES, MessageImpact

IMPACT_FIELDS = (
    "topic_relevance",
    "novelty",
    "continuity",
    "disagreement_strength",
    "social_signal",
    "urgency",
    "event_intensity",
    "engagement_request",
    "topic_shift",
    "primary_topic",
    "secondary_topic",
)

SCORE_FIELDS = IMPACT_FIELDS[:-2]

MESSAGE_IMPACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **{
            field: {"type": "integer", "minimum": 0, "maximum": 100}
            for field in SCORE_FIELDS
        },
        "primary_topic": {"type": "string", "enum": [*TOPIC_CATEGORIES, NO_TOPIC]},
        "secondary_topic": {"type": "string", "enum": [*TOPIC_CATEGORIES, NO_TOPIC]},
    },
    "required": list(IMPACT_FIELDS),
    "additionalProperties": False,
}

EVALUATOR_SYSTEM_PROMPT = """\
Analiza exclusivamente el impacto conversacional del ultimo mensaje del usuario.
No respondas al usuario, no aconsejes y no decidas cambios del estado interno.

Devuelve un objeto JSON que cumpla exactamente el esquema solicitado. Cada valor
es un entero entre 0 y 100:

- topic_relevance: relevancia respecto del tema que se venia tratando.
- novelty: cantidad de informacion o enfoque nuevo.
- continuity: grado en que desarrolla naturalmente el hilo previo.
- disagreement_strength: intensidad del desacuerdo con lo dicho anteriormente;
  usa 0 cuando no exista un desacuerdo identificable.
- social_signal: peso de saludo, cuidado relacional, humor o charla social.
- urgency: necesidad objetiva de una respuesta inmediata.
- event_intensity: importancia del acontecimiento dentro de la conversacion.
- engagement_request: atencion, extension o trabajo que solicita el mensaje.
- topic_shift: grado en que propone cambiar el tema actual.

Clasifica además el contenido, sin decidir si resulta interesante para Nora:

- primary_topic: categoria principal del mensaje.
- secondary_topic: segunda categoria solo si es claramente relevante; usa "none" si no
  existe. Usa "none" en ambos campos para saludos o mensajes sin contenido tematico.

Categorias disponibles:

- everyday_life: vida cotidiana, rutinas y charla general.
- personal_social: relaciones, experiencias personales y vida social.
- arts_literature: arte, literatura, musica, cine y otras obras culturales.
- philosophy_ideas: filosofia, ideas abstractas, conciencia y lenguaje.
- technology_science: tecnologia, informatica, IA, ciencias y matematica.
- politics_society: politica, instituciones, economia y cuestiones sociales.
- work_study: trabajo, educacion, aprendizaje y proyectos.
- practical_hobbies: cocina, bricolaje, deportes y actividades practicas.
- travel_nature_weather: viajes, lugares, naturaleza, animales y clima.

Usa el contexto para interpretar continuidad, relevancia y desacuerdo, y para resolver
referencias tematicas como "eso", "sigamos con lo anterior" o "contame mas". En esos
casos conserva la categoria del tema referido. Si el usuario delega la eleccion de un
tema nuevo sin proponer ninguno, usa "none": el generador elegira segun las preferencias
de Nora. No infieras emociones, diagnosticos ni intenciones no expresadas.
No incluyas explicaciones, deltas sugeridos ni campos adicionales.
"""


class ImpactParseError(ValueError):
    """Raised when evaluator output cannot be trusted by the state engine."""


def parse_message_impact(
    payload: str | Mapping[str, Any],
    *,
    allow_legacy: bool = False,
) -> MessageImpact:
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ImpactParseError("El evaluador no devolvio JSON valido") from exc
    elif isinstance(payload, Mapping):
        decoded = dict(payload)
    else:
        raise ImpactParseError("La salida del evaluador debe ser un objeto JSON")

    if not isinstance(decoded, dict):
        raise ImpactParseError("La salida del evaluador debe ser un objeto JSON")

    received = set(decoded)
    expected = set(IMPACT_FIELDS)
    legacy_fields = set(SCORE_FIELDS)
    if allow_legacy and received == legacy_fields:
        decoded["primary_topic"] = NO_TOPIC
        decoded["secondary_topic"] = NO_TOPIC
        received = set(decoded)
    missing = sorted(expected - received)
    extra = sorted(received - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"faltan: {', '.join(missing)}")
        if extra:
            details.append(f"sobran: {', '.join(extra)}")
        raise ImpactParseError("Salida incompatible (" + "; ".join(details) + ")")

    try:
        return MessageImpact(**decoded)
    except (TypeError, ValueError) as exc:
        raise ImpactParseError(f"Valores de evaluacion invalidos: {exc}") from exc

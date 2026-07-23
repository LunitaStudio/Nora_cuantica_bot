from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


def _plain(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(character for character in normalized if unicodedata.category(character) != "Mn")


def _contains(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


class DemoEvaluatorModel:
    """Deterministic local evaluator for UI development, not experiments."""

    provider = "demo_local"
    model = "heuristic-evaluator-v1"

    async def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        del system_prompt, response_schema
        text = messages[-1]["content"]
        plain = _plain(text)
        has_history = len(messages) > 1
        disagreement = 65 if _contains(
            plain,
            ("no estoy de acuerdo", "discrepo", "pero no", "te equivocas", "incorrecto"),
        ) else 0
        social = 75 if _contains(
            plain,
            ("hola", "buen dia", "buenas", "como estas", "gracias", "jaja"),
        ) else 20
        urgency = 90 if _contains(
            plain,
            ("urgente", "ahora mismo", "emergencia", "peligro", "ayuda ya"),
        ) else 10
        intensity = 82 if _contains(
            plain,
            ("abandonar", "renunciar", "murio", "despedido", "crisis", "terminar todo"),
        ) else min(65, 20 + len(text) // 8)
        topic_shift = 90 if _contains(
            plain,
            ("cambiando de tema", "otra cosa", "dejando eso", "por otro lado"),
        ) else (20 if has_history else 0)
        topic = "everyday_life"
        if _contains(plain, ("borges", "libro", "musica", "pelicula", "arte")):
            topic = "arts_literature"
        elif _contains(plain, ("filosof", "conciencia", "lenguaje")):
            topic = "philosophy_ideas"
        elif _contains(plain, ("inteligencia artificial", " ia", "codigo", "ciencia")):
            topic = "technology_science"
        elif _contains(plain, ("clima", "viaje", "naturaleza", "patagonia")):
            topic = "travel_nature_weather"
        payload = {
            "topic_relevance": 70 if has_history and topic_shift < 50 else 50,
            "novelty": min(90, 45 + len(set(plain.split()))),
            "continuity": 72 if has_history and topic_shift < 50 else 50,
            "disagreement_strength": disagreement,
            "social_signal": social,
            "urgency": urgency,
            "event_intensity": intensity,
            "engagement_request": min(95, 30 + len(text) // 5 + (20 if "?" in text else 0)),
            "topic_shift": topic_shift,
            "primary_topic": topic,
            "secondary_topic": "none",
        }
        return json.dumps(payload)


class DemoGeneratorModel:
    """Small state-aware simulator used only to exercise the complete product locally."""

    provider = "demo_local"
    model = "state-aware-simulator-v1"

    async def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        del response_schema
        match = re.search(
            r"Estado conversacional estructurado:\s*(\{.*?\})\s*Traduccion conductual:",
            system_prompt,
            flags=re.DOTALL,
        )
        state = json.loads(match.group(1)) if match else {}
        user_message = messages[-1]["content"].strip()
        plain = _plain(user_message)

        if _contains(plain, ("ganas de hablar", "para hablar", "queres hablar")):
            availability = state.get("availability", 50)
            interest = state.get("current_interest", 50)
            if availability <= 39:
                core = "Más o menos. Estoy para una charla tranquila, pero no demasiado larga."
            elif availability >= 61 and interest >= 61:
                core = "Sí, hoy estoy bastante para hablar. Me da curiosidad ver por dónde deriva."
            else:
                core = "Sí, aunque más para una charla sin demasiada agenda."
        elif _contains(plain, ("abandonar", "renunciar")):
            core = (
                "Antes de tomar esa decisión, separaría el cansancio del momento de los problemas "
                "estructurales del proyecto. ¿Qué cambió para que abandonar aparezca ahora "
                "como opción?"
            )
        elif _contains(plain, ("hola", "buenas", "buen dia")):
            core = "Hola. ¿Cómo venís?"
        elif "?" in user_message:
            core = (
                "La pregunta merece distinguir supuestos, evidencia y el resultado que querés "
                "obtener. "
                "Con ese marco podemos llegar a una respuesta más precisa."
            )
        else:
            excerpt = user_message if len(user_message) <= 100 else user_message[:97] + "…"
            core = (
                f"Entiendo el punto que planteás sobre «{excerpt}». Conviene mirar qué implica "
                "en la práctica."
            )

        availability = state.get("availability", 50)
        interest = state.get("current_interest", 50)
        if availability >= 61 and interest >= 61:
            core += (
                " Podemos revisar qué lo favorece, qué lo complica y cuál sería la prueba "
                "más pequeña "
                "para salir de la especulación."
            )
        elif availability <= 39:
            core = core.split(". ")[0].rstrip(".") + "."
        return core

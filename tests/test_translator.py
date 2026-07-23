from dataclasses import replace

from nora_quantica.domain.models import TOPIC_CATEGORIES
from nora_quantica.domain.translator import (
    band,
    translate_state,
    translate_topic_personality,
)
from nora_quantica.prompt_profiles.baseline import BASELINE_PROFILE

from .test_engine import make_state


def test_band_boundaries() -> None:
    assert band(0) == "very_low"
    assert band(19) == "very_low"
    assert band(20) == "low"
    assert band(39) == "low"
    assert band(40) == "medium"
    assert band(60) == "medium"
    assert band(61) == "high"
    assert band(80) == "high"
    assert band(81) == "very_high"
    assert band(100) == "very_high"


def test_translation_never_omits_safety_floor_or_reveals_numbers() -> None:
    translation = translate_state(make_state())
    assert BASELINE_PROFILE.safety_floor in translation
    assert "72" not in translation
    assert "estado" in translation


def test_topic_personality_summarizes_extremes_without_numbers() -> None:
    state = make_state()
    affinities = {category: 50 for category in TOPIC_CATEGORIES}
    affinities["travel_nature_weather"] = 96
    affinities["practical_hobbies"] = 84
    affinities["technology_science"] = 3
    state.baseline = replace(state.baseline, topic_affinities=affinities)

    summary = translate_topic_personality(state)

    assert "viajes, naturaleza y clima" in summary
    assert "actividades prácticas y hobbies" in summary
    assert "tecnología y ciencia" in summary
    assert "96" not in summary and "84" not in summary and "3" not in summary

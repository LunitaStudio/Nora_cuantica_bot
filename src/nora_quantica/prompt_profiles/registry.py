from __future__ import annotations

from nora_quantica.prompt_profiles import PromptProfile
from nora_quantica.prompt_profiles.baseline import BASELINE_PROFILE
from nora_quantica.prompt_profiles.experimental import EXPERIMENTAL_PROFILE

PROFILES = {
    BASELINE_PROFILE.name: BASELINE_PROFILE,
    EXPERIMENTAL_PROFILE.name: EXPERIMENTAL_PROFILE,
}


def get_prompt_profile(name: str) -> PromptProfile:
    normalized = name.strip().lower()
    try:
        return PROFILES[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"Perfil desconocido '{name}'. Disponibles: {available}") from exc


def profile_from_snapshot(snapshot: dict) -> PromptProfile:
    return PromptProfile.from_snapshot(snapshot)


__all__ = ["PROFILES", "get_prompt_profile", "profile_from_snapshot"]

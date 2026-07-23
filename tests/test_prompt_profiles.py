from copy import deepcopy

import pytest

from nora_quantica.prompt_profiles import PromptProfile
from nora_quantica.prompt_profiles.baseline import BASELINE_PROFILE
from nora_quantica.prompt_profiles.experimental import EXPERIMENTAL_PROFILE
from nora_quantica.prompt_profiles.registry import get_prompt_profile


def test_registry_selects_both_profiles() -> None:
    assert get_prompt_profile("baseline") is BASELINE_PROFILE
    assert get_prompt_profile("EXPERIMENTAL") is EXPERIMENTAL_PROFILE
    with pytest.raises(ValueError, match="Perfil desconocido"):
        get_prompt_profile("missing")


def test_profiles_have_independent_instruction_copies() -> None:
    assert BASELINE_PROFILE.instructions is not EXPERIMENTAL_PROFILE.instructions
    assert BASELINE_PROFILE.instructions["availability"] is not (
        EXPERIMENTAL_PROFILE.instructions["availability"]
    )


def test_snapshot_round_trip_and_hash_validation() -> None:
    snapshot = BASELINE_PROFILE.snapshot()
    restored = PromptProfile.from_snapshot(snapshot)
    assert restored.content_hash() == BASELINE_PROFILE.content_hash()

    tampered = deepcopy(snapshot)
    tampered["instructions"]["candor"]["high"] += " alterado"
    with pytest.raises(ValueError, match="no coincide con su hash"):
        PromptProfile.from_snapshot(tampered)


def test_profile_requires_every_variable_and_band() -> None:
    instructions = deepcopy(BASELINE_PROFILE.instructions)
    del instructions["availability"]["low"]
    with pytest.raises(ValueError, match="Bandas incompletas"):
        PromptProfile("broken", "1", "prompt", instructions, "safety")

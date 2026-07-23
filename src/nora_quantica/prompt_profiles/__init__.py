from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

PROFILE_VARIABLES = (
    "relational_closeness",
    "availability",
    "current_interest",
    "candor",
    "topic_orientation",
)

PROFILE_BANDS = ("very_low", "low", "medium", "high", "very_high")


@dataclass(frozen=True, slots=True)
class PromptProfile:
    name: str
    version: str
    base_prompt: str
    instructions: dict[str, dict[str, str]]
    safety_floor: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", deepcopy(self.instructions))
        if not self.name.strip() or not self.version.strip():
            raise ValueError("El perfil necesita nombre y version")
        if not self.base_prompt.strip() or not self.safety_floor.strip():
            raise ValueError("El perfil necesita prompt base y piso de seguridad")
        if set(self.instructions) != set(PROFILE_VARIABLES):
            raise ValueError("Las variables del perfil no coinciden con el estado conversacional")
        for variable, bands in self.instructions.items():
            if set(bands) != set(PROFILE_BANDS):
                raise ValueError(f"Bandas incompletas para {variable}")
            if any(not isinstance(text, str) or not text.strip() for text in bands.values()):
                raise ValueError(f"Todas las instrucciones de {variable} deben contener texto")

    def content_hash(self) -> str:
        payload = {
            "base_prompt": self.base_prompt,
            "instructions": self.instructions,
            "safety_floor": self.safety_floor,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "content_hash": self.content_hash(),
            "base_prompt": self.base_prompt,
            "instructions": deepcopy(self.instructions),
            "safety_floor": self.safety_floor,
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> PromptProfile:
        profile = cls(
            name=snapshot["name"],
            version=snapshot["version"],
            base_prompt=snapshot["base_prompt"],
            instructions=deepcopy(snapshot["instructions"]),
            safety_floor=snapshot["safety_floor"],
        )
        recorded_hash = snapshot.get("content_hash")
        if recorded_hash and recorded_hash != profile.content_hash():
            raise ValueError("El contenido del perfil no coincide con su hash")
        return profile


__all__ = ["PROFILE_BANDS", "PROFILE_VARIABLES", "PromptProfile"]

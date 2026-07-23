from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # The core domain does not require the optional API dependency.
    load_dotenv = None


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} debe ser numerico") from exc


@dataclass(frozen=True, slots=True)
class ModelSettings:
    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float


@dataclass(frozen=True, slots=True)
class Settings:
    app_mode: str
    prompt_profile: str
    database_path: str
    anu_qrng_api_key: str | None
    anu_qrng_url: str
    evaluator: ModelSettings
    generator: ModelSettings
    behavior_policy: str = "moves_v1"
    history_limit: int = 24
    experiment_initial_bytes_hex: str | None = None
    experiment_behavior_seed_id: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        if load_dotenv is not None:
            load_dotenv()
        shared_key = os.getenv("LLM_API_KEY", "")
        shared_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        return cls(
            app_mode=os.getenv("APP_MODE", "demo").lower(),
            prompt_profile=os.getenv("PROMPT_PROFILE", "baseline").lower(),
            database_path=os.getenv("DATABASE_PATH", "data/nora_quantica.db"),
            anu_qrng_api_key=os.getenv("ANU_QRNG_API_KEY"),
            anu_qrng_url=os.getenv("ANU_QRNG_URL", ANU_DEFAULT),
            evaluator=ModelSettings(
                provider=os.getenv("EVALUATOR_PROVIDER", "openai_compatible"),
                model=os.getenv("EVALUATOR_MODEL", "gpt-4.1-mini"),
                api_key=os.getenv("EVALUATOR_API_KEY", shared_key),
                base_url=os.getenv("EVALUATOR_BASE_URL", shared_url),
                temperature=_float_env("EVALUATOR_TEMPERATURE", 0.0),
            ),
            generator=ModelSettings(
                provider=os.getenv("GENERATOR_PROVIDER", "openai_compatible"),
                model=os.getenv("GENERATOR_MODEL", "gpt-4.1-mini"),
                api_key=os.getenv("GENERATOR_API_KEY", shared_key),
                base_url=os.getenv("GENERATOR_BASE_URL", shared_url),
                temperature=_float_env("GENERATOR_TEMPERATURE", 0.35),
            ),
            behavior_policy=os.getenv("BEHAVIOR_POLICY", "moves_v1").lower(),
            history_limit=int(os.getenv("HISTORY_LIMIT", "24")),
            experiment_initial_bytes_hex=(
                os.getenv("EXPERIMENT_INITIAL_BYTES_HEX", "").strip() or None
            ),
            experiment_behavior_seed_id=(
                os.getenv("EXPERIMENT_BEHAVIOR_SEED_ID", "").strip() or None
            ),
        )


ANU_DEFAULT = "https://api.quantumnumbers.anu.edu.au"

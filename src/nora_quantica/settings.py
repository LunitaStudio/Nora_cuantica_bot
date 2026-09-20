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
    max_turns: int = 24
    storage_backend: str = "sqlite"
    firestore_project_id: str | None = None
    firestore_collection_prefix: str = "nora"
    conversation_retention_hours: int = 48
    rate_limit_enabled: bool = False
    conversations_per_hour: int = 3
    conversations_per_ip_day: int = 10
    messages_per_minute: int = 12
    messages_per_ip_day: int = 120
    session_secret: str | None = None
    session_cookie_secure: bool = False
    lab_detail_level: str = "full"
    session_export_enabled: bool = True
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
            max_turns=int(os.getenv("MAX_TURNS_PER_CONVERSATION", "24")),
            storage_backend=os.getenv("STORAGE_BACKEND", "sqlite").lower(),
            firestore_project_id=os.getenv("FIRESTORE_PROJECT_ID") or None,
            firestore_collection_prefix=os.getenv(
                "FIRESTORE_COLLECTION_PREFIX", "nora"
            ),
            conversation_retention_hours=int(
                os.getenv("CONVERSATION_RETENTION_HOURS", "48")
            ),
            rate_limit_enabled=os.getenv("RATE_LIMIT_ENABLED", "false").lower()
            in {"1", "true", "yes"},
            conversations_per_hour=int(
                os.getenv("CONVERSATIONS_PER_HOUR", "3")
            ),
            conversations_per_ip_day=int(
                os.getenv("CONVERSATIONS_PER_IP_DAY", "10")
            ),
            messages_per_minute=int(os.getenv("MESSAGES_PER_MINUTE", "12")),
            messages_per_ip_day=int(os.getenv("MESSAGES_PER_IP_DAY", "120")),
            session_secret=os.getenv("SESSION_SECRET") or None,
            session_cookie_secure=os.getenv(
                "SESSION_COOKIE_SECURE", "false"
            ).lower()
            in {"1", "true", "yes"},
            lab_detail_level=os.getenv("LAB_DETAIL_LEVEL", "full").lower(),
            session_export_enabled=os.getenv(
                "SESSION_EXPORT_ENABLED", "true"
            ).lower()
            in {"1", "true", "yes"},
            experiment_initial_bytes_hex=(
                os.getenv("EXPERIMENT_INITIAL_BYTES_HEX", "").strip() or None
            ),
            experiment_behavior_seed_id=(
                os.getenv("EXPERIMENT_BEHAVIOR_SEED_ID", "").strip() or None
            ),
        )


ANU_DEFAULT = "https://api.quantumnumbers.anu.edu.au"

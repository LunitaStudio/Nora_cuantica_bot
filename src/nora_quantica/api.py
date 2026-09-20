from __future__ import annotations

import hashlib
import hmac
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi import Path as ApiPath
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nora_quantica.application.conversation_service import (
    MAX_MESSAGE_LENGTH,
    ConversationNotFoundError,
    ConversationService,
    ConversationTurnLimitError,
)
from nora_quantica.domain.evaluator import ImpactParseError
from nora_quantica.infrastructure.chat_models import ModelProviderError, OpenAICompatibleChatModel
from nora_quantica.infrastructure.demo_models import DemoEvaluatorModel, DemoGeneratorModel
from nora_quantica.infrastructure.entropy import (
    AnuQuantumProvider,
    EntropyBroker,
    FixedEntropyAcquirer,
)
from nora_quantica.infrastructure.firestore_store import FirestoreStore
from nora_quantica.infrastructure.rate_limit import (
    FirestoreRateLimiter,
    InMemoryRateLimiter,
    RateLimiter,
)
from nora_quantica.infrastructure.sqlite_store import SQLiteStore
from nora_quantica.prompt_profiles.registry import get_prompt_profile
from nora_quantica.settings import ModelSettings, Settings


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


SESSION_COOKIE_NAME = "__session"


class RateLimitExceeded(RuntimeError):
    def __init__(self, retry_after: int) -> None:
        super().__init__("Demasiadas solicitudes. Probá de nuevo en unos minutos.")
        self.retry_after = retry_after


def _sign_session(session_id: str, secret: bytes) -> str:
    signature = hmac.new(secret, session_id.encode(), hashlib.sha256).hexdigest()
    return f"{session_id}.{signature}"


def _session_id(token: str | None, secret: bytes) -> tuple[str, bool]:
    if token:
        session_id, separator, signature = token.partition(".")
        expected = hmac.new(secret, session_id.encode(), hashlib.sha256).hexdigest()
        if (
            separator
            and len(session_id) >= 24
            and hmac.compare_digest(signature, expected)
        ):
            return session_id, False
    return secrets.token_urlsafe(24), True


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def _live_model(config: ModelSettings) -> OpenAICompatibleChatModel:
    if config.provider != "openai_compatible":
        raise ValueError(
            f"Proveedor no soportado en este MVP: {config.provider}. "
            "Use 'openai_compatible' o APP_MODE=demo."
        )
    return OpenAICompatibleChatModel(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
    )


def build_service(settings: Settings) -> ConversationService:
    if settings.storage_backend == "sqlite":
        store = SQLiteStore(settings.database_path)
    elif settings.storage_backend == "firestore":
        store = FirestoreStore(
            project_id=settings.firestore_project_id,
            collection_prefix=settings.firestore_collection_prefix,
            retention_hours=settings.conversation_retention_hours,
        )
    else:
        raise ValueError(
            f"STORAGE_BACKEND no soportado: {settings.storage_backend}"
        )
    store.initialize()
    if settings.experiment_initial_bytes_hex is not None:
        entropy = FixedEntropyAcquirer(settings.experiment_initial_bytes_hex)
    else:
        entropy = EntropyBroker(
            AnuQuantumProvider(
                settings.anu_qrng_api_key,
                endpoint=settings.anu_qrng_url,
            ),
            store,
        )
    if settings.app_mode == "demo":
        evaluator = DemoEvaluatorModel()
        generator = DemoGeneratorModel()
    elif settings.app_mode == "live":
        evaluator = _live_model(settings.evaluator)
        generator = _live_model(settings.generator)
    else:
        raise ValueError("APP_MODE debe ser 'demo' o 'live'")
    return ConversationService(
        store=store,
        entropy=entropy,
        evaluator=evaluator,
        generator=generator,
        history_limit=settings.history_limit,
        max_turns=settings.max_turns,
        prompt_profile=get_prompt_profile(settings.prompt_profile),
        behavior_policy=settings.behavior_policy,
        behavior_seed_id=settings.experiment_behavior_seed_id,
    )


def create_app(
    service: ConversationService | None = None,
    settings: Settings | None = None,
    rate_limiter: RateLimiter | None = None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    if app_settings.lab_detail_level not in {"simple", "full"}:
        raise ValueError("LAB_DETAIL_LEVEL debe ser 'simple' o 'full'")
    app_service = service or build_service(app_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        app_service.store.initialize()
        yield

    app = FastAPI(
        title="Nora Quantica",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.conversation_service = app_service
    app.state.settings = app_settings
    if rate_limiter is not None:
        app_limiter = rate_limiter
    elif isinstance(app_service.store, FirestoreStore):
        app_limiter = FirestoreRateLimiter(
            app_service.store.client,
            collection=f"{app_settings.firestore_collection_prefix}_rate_limits",
        )
    else:
        app_limiter = InMemoryRateLimiter()
    app.state.rate_limiter = app_limiter
    session_secret = (
        app_settings.session_secret.encode()
        if app_settings.session_secret
        else secrets.token_bytes(32)
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        session_id, is_new_session = _session_id(
            request.cookies.get(SESSION_COOKIE_NAME),
            session_secret,
        )
        request.state.session_id = session_id
        request.state.session_identity = hmac.new(
            session_secret,
            session_id.encode(),
            hashlib.sha256,
        ).hexdigest()
        request.state.ip_identity = hmac.new(
            session_secret,
            _client_ip(request).encode(),
            hashlib.sha256,
        ).hexdigest()
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        if is_new_session:
            response.set_cookie(
                SESSION_COOKIE_NAME,
                _sign_session(session_id, session_secret),
                max_age=60 * 60 * 24 * 30,
                httponly=True,
                secure=app_settings.session_cookie_secure,
                samesite="lax",
            )
        return response

    def enforce_limit(
        request: Request,
        bucket: str,
        identity: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> None:
        if not app_settings.rate_limit_enabled:
            return
        result = app_limiter.hit(
            bucket,
            identity,
            limit=limit,
            window_seconds=window_seconds,
        )
        if not result.allowed:
            raise RateLimitExceeded(result.retry_after)

    @app.exception_handler(ConversationNotFoundError)
    async def not_found_handler(_, exc: ConversationNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"detail": f"Conversación inexistente: {exc.args[0]}"},
        )

    @app.exception_handler(ConversationTurnLimitError)
    async def turn_limit_handler(_, exc: ConversationTurnLimitError):
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(_, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"detail": str(exc)},
            headers={"Retry-After": str(exc.retry_after)},
        )

    @app.exception_handler(ModelProviderError)
    async def provider_error_handler(_, exc: ModelProviderError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ImpactParseError)
    async def evaluator_error_handler(_, exc: ImpactParseError):
        return JSONResponse(
            status_code=502,
            content={"detail": f"La evaluación estructurada falló: {exc}"},
        )

    @app.get("/api/health")
    async def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "mode": app_settings.app_mode,
            "lab_detail_level": app_settings.lab_detail_level,
            "session_export_enabled": app_settings.session_export_enabled,
        }

    @app.post("/api/conversations", status_code=201)
    async def create_conversation(request: Request) -> dict[str, Any]:
        enforce_limit(
            request,
            "conversation-session-hour",
            request.state.session_id,
            limit=app_settings.conversations_per_hour,
            window_seconds=60 * 60,
        )
        enforce_limit(
            request,
            "conversation-ip-day",
            request.state.ip_identity,
            limit=app_settings.conversations_per_ip_day,
            window_seconds=60 * 60 * 24,
        )
        state = await app_service.create_conversation(request.state.session_identity)
        return {
            "conversation_id": state.conversation_id,
            "mode": app_settings.app_mode,
            "prompt_profile": state.metadata["prompt_profile"]["name"],
            "behavior_policy": state.metadata["behavior_policy"]["name"],
        }

    @app.get("/api/conversations/{conversation_id}/messages")
    async def messages(
        request: Request,
        conversation_id: Annotated[str, ApiPath(min_length=1)],
    ) -> dict[str, Any]:
        app_service.assert_owner(
            conversation_id,
            request.state.session_identity,
        )
        items = app_service.messages(conversation_id)
        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "role": item.role,
                    "content": item.content,
                    "turn_number": item.turn_number,
                    "created_at": item.created_at.isoformat(),
                }
                for item in items
            ],
        }

    @app.post("/api/conversations/{conversation_id}/messages")
    async def send_message(
        request: Request,
        conversation_id: Annotated[str, ApiPath(min_length=1)],
        payload: MessageRequest,
    ) -> dict[str, Any]:
        app_service.assert_owner(
            conversation_id,
            request.state.session_identity,
        )
        enforce_limit(
            request,
            "message-session-minute",
            request.state.session_id,
            limit=app_settings.messages_per_minute,
            window_seconds=60,
        )
        enforce_limit(
            request,
            "message-ip-day",
            request.state.ip_identity,
            limit=app_settings.messages_per_ip_day,
            window_seconds=60 * 60 * 24,
        )
        try:
            result = await app_service.send_message(conversation_id, payload.message)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "conversation_id": conversation_id,
            "response": result.response,
            "turn_number": result.turn_number,
        }

    @app.get("/api/conversations/{conversation_id}/lab")
    async def lab(
        request: Request,
        conversation_id: Annotated[str, ApiPath(min_length=1)],
    ) -> dict[str, Any]:
        app_service.assert_owner(
            conversation_id,
            request.state.session_identity,
        )
        snapshot = app_service.laboratory_snapshot(conversation_id)
        if app_settings.lab_detail_level == "full":
            return snapshot

        source = snapshot["quantum_source"]
        public_source = {
            "provider": source["provider"],
            "retrieved_at": source["retrieved_at"],
            "fallback_used": source["fallback_used"],
        }
        last_turn = snapshot["last_turn"]
        public_turn = None
        if last_turn is not None:
            public_turn = {
                "turn_number": last_turn["turn_number"],
                "impact": last_turn["impact"],
                "effective_deltas": last_turn["effective_deltas"],
            }
        return {
            "conversation_id": snapshot["conversation_id"],
            "turn_number": snapshot["turn_number"],
            "quantum_source": public_source,
            "baseline": snapshot["baseline"],
            "current": snapshot["current"],
            "last_topic_affinity": snapshot["last_topic_affinity"],
            "last_turn": public_turn,
        }

    @app.get("/api/conversations/{conversation_id}/export")
    async def export(
        request: Request,
        conversation_id: Annotated[str, ApiPath(min_length=1)],
    ) -> JSONResponse:
        if not app_settings.session_export_enabled:
            raise HTTPException(status_code=404, detail="Recurso no disponible")
        app_service.assert_owner(
            conversation_id,
            request.state.session_identity,
        )
        return JSONResponse(
            content=app_service.export_session(conversation_id),
            headers={
                "Content-Disposition": f'attachment; filename="nora-{conversation_id}.json"'
            },
        )

    web_dir = Path(__file__).resolve().parent / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(web_dir / "index.html")

    return app


app = create_app()

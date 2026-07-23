from __future__ import annotations

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
)
from nora_quantica.domain.evaluator import ImpactParseError
from nora_quantica.infrastructure.chat_models import ModelProviderError, OpenAICompatibleChatModel
from nora_quantica.infrastructure.demo_models import DemoEvaluatorModel, DemoGeneratorModel
from nora_quantica.infrastructure.entropy import (
    AnuQuantumProvider,
    EntropyBroker,
    FixedEntropyAcquirer,
)
from nora_quantica.infrastructure.sqlite_store import SQLiteStore
from nora_quantica.prompt_profiles.registry import get_prompt_profile
from nora_quantica.settings import ModelSettings, Settings


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


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
    store = SQLiteStore(settings.database_path)
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
        prompt_profile=get_prompt_profile(settings.prompt_profile),
        behavior_policy=settings.behavior_policy,
        behavior_seed_id=settings.experiment_behavior_seed_id,
    )


def create_app(
    service: ConversationService | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
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

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
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
        return response

    @app.exception_handler(ConversationNotFoundError)
    async def not_found_handler(_, exc: ConversationNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"detail": f"Conversación inexistente: {exc.args[0]}"},
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
    async def health() -> dict[str, str]:
        return {"status": "ok", "mode": app_settings.app_mode}

    @app.post("/api/conversations", status_code=201)
    async def create_conversation() -> dict[str, Any]:
        state = await app_service.create_conversation()
        return {
            "conversation_id": state.conversation_id,
            "mode": app_settings.app_mode,
            "prompt_profile": state.metadata["prompt_profile"]["name"],
            "behavior_policy": state.metadata["behavior_policy"]["name"],
        }

    @app.get("/api/conversations/{conversation_id}/messages")
    async def messages(
        conversation_id: Annotated[str, ApiPath(min_length=1)],
    ) -> dict[str, Any]:
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
        conversation_id: Annotated[str, ApiPath(min_length=1)],
        payload: MessageRequest,
    ) -> dict[str, Any]:
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
        conversation_id: Annotated[str, ApiPath(min_length=1)],
    ) -> dict[str, Any]:
        return app_service.laboratory_snapshot(conversation_id)

    @app.get("/api/conversations/{conversation_id}/export")
    async def export(
        conversation_id: Annotated[str, ApiPath(min_length=1)],
    ) -> JSONResponse:
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

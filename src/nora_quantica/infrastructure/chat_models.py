from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelProviderError(RuntimeError):
    pass


class OpenAICompatibleChatModel:
    """Minimal provider-neutral adapter for OpenAI-compatible chat completions APIs."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.3,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError(f"Falta API key para {provider}")
        self._provider = provider
        self._model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._request,
            system_prompt,
            messages,
            response_schema,
        )

    def _request(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": self.temperature,
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "message_impact",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Nora-Quantica/0.1",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                decoded = json.loads(response.read())
            content = decoded["choices"][0]["message"]["content"]
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise ModelProviderError(f"Fallo de red en {self.provider}: {exc}") from exc
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ModelProviderError(f"Respuesta invalida de {self.provider}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelProviderError(f"{self.provider} devolvio contenido vacio")
        return content.strip()

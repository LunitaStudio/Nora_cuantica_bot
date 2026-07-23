from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from nora_quantica.application.ports import EntropyCache
from nora_quantica.domain.entropy import (
    INITIAL_STATE_BYTES,
    EntropyProvider,
    EntropySample,
    OsCsprngProvider,
)
from nora_quantica.domain.models import EntropyMetadata

ANU_API_URL = "https://api.quantumnumbers.anu.edu.au"


class QuantumProviderError(RuntimeError):
    pass


class FixedEntropyAcquirer:
    """Repeatable state source for controlled experiments; never presented as QRNG."""

    def __init__(self, raw_bytes_hex: str) -> None:
        try:
            raw_bytes = bytes.fromhex(raw_bytes_hex)
        except ValueError as exc:
            raise ValueError("EXPERIMENT_INITIAL_BYTES_HEX debe ser hexadecimal valido") from exc
        if len(raw_bytes) < INITIAL_STATE_BYTES:
            raise ValueError(
                "EXPERIMENT_INITIAL_BYTES_HEX requiere al menos "
                f"{INITIAL_STATE_BYTES} bytes ({INITIAL_STATE_BYTES * 2} caracteres hex)"
            )
        self.raw_bytes = raw_bytes

    async def acquire(self) -> EntropySample:
        digest = hashlib.sha256(self.raw_bytes).hexdigest()
        return EntropySample(
            raw_bytes=self.raw_bytes,
            metadata=EntropyMetadata(
                provider="controlled_experiment_fixture",
                retrieved_at=datetime.now(UTC),
                raw_bytes_hash=digest,
                response_id=f"fixture-{digest[:16]}",
            ),
        )


class AnuQuantumProvider:
    """Client for ANU Quantum Numbers' uint8 JSON API."""

    def __init__(
        self,
        api_key: str | None,
        *,
        endpoint: str = ANU_API_URL,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    async def get_bytes(self, count: int) -> EntropySample:
        if not 1 <= count <= 1024:
            raise ValueError("ANU admite entre 1 y 1024 bytes por solicitud")
        if not self.api_key:
            raise QuantumProviderError("Falta ANU_QRNG_API_KEY")
        return await asyncio.to_thread(self._request, count)

    def _request(self, count: int) -> EntropySample:
        url = f"{self.endpoint}?{urlencode({'length': count, 'type': 'uint8'})}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Nora-Quantica/0.1",
                "x-api-key": self.api_key or "",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                raw_payload = response.read()
                response_id = response.headers.get("x-amzn-requestid") or response.headers.get(
                    "x-request-id"
                )
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise QuantumProviderError(f"ANU QRNG no disponible: {exc}") from exc

        try:
            payload: Any = json.loads(raw_payload)
            values = payload["data"]
            success = payload.get("success", True)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise QuantumProviderError("Respuesta invalida de ANU QRNG") from exc
        if success is not True or not isinstance(values, list) or len(values) != count:
            raise QuantumProviderError("ANU QRNG devolvio una cantidad o estado invalido")
        invalid_values = any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255
            for value in values
        )
        if invalid_values:
            raise QuantumProviderError("ANU QRNG devolvio valores fuera de uint8")

        raw_bytes = bytes(values)
        return EntropySample(
            raw_bytes=raw_bytes,
            metadata=EntropyMetadata(
                provider="anu_quantum_numbers",
                retrieved_at=datetime.now(UTC),
                raw_bytes_hash=hashlib.sha256(raw_bytes).hexdigest(),
                response_id=response_id or hashlib.sha256(raw_payload).hexdigest()[:24],
            ),
        )


class EntropyBroker:
    """Acquire state entropy using QRNG, quantum cache, then OS CSPRNG."""

    def __init__(
        self,
        primary: EntropyProvider,
        cache: EntropyCache,
        fallback: EntropyProvider | None = None,
        *,
        state_bytes: int = INITIAL_STATE_BYTES,
        batch_size: int = 32,
    ) -> None:
        if state_bytes < INITIAL_STATE_BYTES:
            raise ValueError(
                f"El estado inicial requiere al menos {INITIAL_STATE_BYTES} bytes"
            )
        if batch_size < state_bytes:
            raise ValueError("El lote no puede ser menor que los bytes de estado")
        self.primary = primary
        self.cache = cache
        self.fallback = fallback or OsCsprngProvider()
        self.state_bytes = state_bytes
        self.batch_size = batch_size

    async def acquire(self) -> EntropySample:
        try:
            sample = await self.primary.get_bytes(self.batch_size)
            if len(sample.raw_bytes) < self.state_bytes:
                raise QuantumProviderError("La fuente primaria devolvio entropia insuficiente")
            remainder = sample.raw_bytes[self.state_bytes :]
            if remainder:
                cached_metadata = replace(
                    sample.metadata,
                    raw_bytes_hash=hashlib.sha256(remainder).hexdigest(),
                )
                self.cache.put(EntropySample(remainder, cached_metadata))
            return sample
        except (QuantumProviderError, HTTPError, URLError, TimeoutError, OSError, ValueError):
            cached = self.cache.take(self.state_bytes)
            if cached is not None:
                return cached
            return await self.fallback.get_bytes(self.batch_size)

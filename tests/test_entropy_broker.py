import asyncio
import hashlib
from datetime import UTC, datetime

from nora_quantica.domain.entropy import INITIAL_STATE_BYTES, EntropySample
from nora_quantica.domain.models import EntropyMetadata
from nora_quantica.infrastructure.entropy import EntropyBroker, QuantumProviderError


def sample(raw: bytes, provider: str = "quantum") -> EntropySample:
    return EntropySample(
        raw,
        EntropyMetadata(
            provider=provider,
            retrieved_at=datetime.now(UTC),
            raw_bytes_hash=hashlib.sha256(raw).hexdigest(),
        ),
    )


class Provider:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    async def get_bytes(self, count: int):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class Cache:
    def __init__(self, cached=None):
        self.cached = cached
        self.put_samples = []

    def put(self, value):
        self.put_samples.append(value)

    def take(self, count):
        value, self.cached = self.cached, None
        return value


def test_broker_prefers_quantum_and_reserves_unused_bytes() -> None:
    primary = Provider(result=sample(bytes(range(32))))
    cache = Cache()
    fallback = Provider(result=sample(b"f" * 32, "os"))
    acquired = asyncio.run(EntropyBroker(primary, cache, fallback).acquire())
    assert acquired.raw_bytes == bytes(range(32))
    assert cache.put_samples[0].raw_bytes == bytes(range(INITIAL_STATE_BYTES, 32))
    assert fallback.calls == 0


def test_broker_uses_cache_before_os_fallback() -> None:
    cached = sample(b"quantum-cache!", "quantum_cache")
    primary = Provider(error=QuantumProviderError("offline"))
    cache = Cache(cached)
    fallback = Provider(result=sample(b"f" * 32, "os"))
    acquired = asyncio.run(EntropyBroker(primary, cache, fallback).acquire())
    assert acquired is cached
    assert fallback.calls == 0


def test_broker_uses_os_when_quantum_and_cache_fail() -> None:
    fallback_sample = sample(b"f" * 32, "os")
    primary = Provider(error=QuantumProviderError("offline"))
    fallback = Provider(result=fallback_sample)
    acquired = asyncio.run(EntropyBroker(primary, Cache(), fallback).acquire())
    assert acquired is fallback_sample
    assert fallback.calls == 1

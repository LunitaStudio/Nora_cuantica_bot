import asyncio
import hashlib
from datetime import UTC, datetime

import pytest

from nora_quantica.domain.entropy import (
    EntropySample,
    OsCsprngProvider,
    byte_to_score,
    initial_state_from_entropy,
)
from nora_quantica.domain.models import TOPIC_CATEGORIES, EntropyMetadata
from nora_quantica.infrastructure.entropy import FixedEntropyAcquirer


def sample(raw: bytes) -> EntropySample:
    return EntropySample(
        raw_bytes=raw,
        metadata=EntropyMetadata(
            provider="test",
            retrieved_at=datetime.now(UTC),
            raw_bytes_hash=hashlib.sha256(raw).hexdigest(),
        ),
    )


def test_byte_to_score_maps_full_range() -> None:
    assert byte_to_score(0) == 0
    assert byte_to_score(255) == 100
    assert all(0 <= byte_to_score(value) <= 100 for value in range(256))
    assert all(
        byte_to_score(value) <= byte_to_score(value + 1) for value in range(255)
    )


@pytest.mark.parametrize("value", [-1, 256])
def test_byte_to_score_rejects_out_of_range(value: int) -> None:
    with pytest.raises(ValueError):
        byte_to_score(value)


def test_initial_state_uses_first_five_bytes_and_keeps_closeness_baseline() -> None:
    raw = bytes([0, 64, 128, 192, 255, *range(9)])
    state = initial_state_from_entropy(sample(raw))
    assert state.baseline.relational_closeness == 0
    assert state.baseline.availability == 25
    assert state.baseline.interest_bias == 50
    assert state.baseline.candor == 75
    assert state.baseline.topic_orientation == 100
    assert state.current.availability == 25
    assert state.baseline.topic_affinities == {
        category: byte_to_score(index) for index, category in enumerate(TOPIC_CATEGORIES)
    }


def test_initial_state_requires_bytes_for_every_topic_affinity() -> None:
    with pytest.raises(ValueError, match="14 bytes"):
        initial_state_from_entropy(sample(bytes(range(13))))


def test_os_csprng_is_explicitly_marked_as_fallback() -> None:
    entropy = asyncio.run(OsCsprngProvider().get_bytes(32))
    assert len(entropy.raw_bytes) == 32
    assert entropy.metadata.fallback_used is True
    assert entropy.metadata.fallback_type == "os_csprng"


def test_fixed_experiment_entropy_is_repeatable_and_explicitly_labeled() -> None:
    source = FixedEntropyAcquirer("000102030405060708090a0b0c0d")

    first = asyncio.run(source.acquire())
    second = asyncio.run(source.acquire())

    assert first.raw_bytes == second.raw_bytes == bytes(range(14))
    assert first.metadata.raw_bytes_hash == second.metadata.raw_bytes_hash
    assert first.metadata.provider == "controlled_experiment_fixture"
    assert first.metadata.fallback_used is False


@pytest.mark.parametrize("value", ["zz", "000102"])
def test_fixed_experiment_entropy_rejects_invalid_or_short_hex(value: str) -> None:
    with pytest.raises(ValueError, match="EXPERIMENT_INITIAL_BYTES_HEX"):
        FixedEntropyAcquirer(value)

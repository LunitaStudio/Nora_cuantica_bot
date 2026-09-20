from nora_quantica.infrastructure.rate_limit import InMemoryRateLimiter


def test_in_memory_rate_limiter_blocks_after_limit() -> None:
    limiter = InMemoryRateLimiter()

    assert limiter.hit("messages", "session", limit=2, window_seconds=60).allowed
    assert limiter.hit("messages", "session", limit=2, window_seconds=60).allowed
    blocked = limiter.hit("messages", "session", limit=2, window_seconds=60)

    assert blocked.allowed is False
    assert blocked.retry_after > 0

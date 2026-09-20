from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    allowed: bool
    retry_after: int


class RateLimiter(Protocol):
    def hit(
        self,
        bucket: str,
        identity: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult: ...


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._counts: dict[tuple[str, str, int], int] = {}
        self._lock = threading.Lock()

    def hit(
        self,
        bucket: str,
        identity: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("Los límites deben ser positivos")
        now = int(time.time())
        window = now // window_seconds
        key = (bucket, identity, window)
        with self._lock:
            count = self._counts.get(key, 0) + 1
            self._counts[key] = count
            if len(self._counts) > 10_000:
                self._counts = {
                    stored_key: value
                    for stored_key, value in self._counts.items()
                    if stored_key[2] >= window - 1
                }
        retry_after = max(1, (window + 1) * window_seconds - now)
        return RateLimitResult(count <= limit, retry_after)


class FirestoreRateLimiter:
    def __init__(
        self,
        client: Any,
        *,
        collection: str = "nora_rate_limits",
    ) -> None:
        self.client = client
        self.collection = client.collection(collection)

    def hit(
        self,
        bucket: str,
        identity: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("Los límites deben ser positivos")
        try:
            from google.cloud import firestore
        except ImportError as exc:
            raise RuntimeError(
                "Rate limiting con Firestore requiere la dependencia 'cloud'"
            ) from exc

        now = int(time.time())
        window = now // window_seconds
        digest = hashlib.sha256(
            f"{bucket}:{identity}:{window}".encode()
        ).hexdigest()
        reference = self.collection.document(digest)
        transaction = self.client.transaction()

        @firestore.transactional
        def increment(current_transaction):
            snapshot = reference.get(transaction=current_transaction)
            count = int(snapshot.get("count")) + 1 if snapshot.exists else 1
            current_transaction.set(
                reference,
                {
                    "bucket": bucket,
                    "window": window,
                    "count": count,
                    "expires_at": datetime.now(UTC)
                    + timedelta(seconds=window_seconds * 2),
                },
            )
            return count

        count = increment(transaction)
        retry_after = max(1, (window + 1) * window_seconds - now)
        return RateLimitResult(count <= limit, retry_after)

from nora_quantica.infrastructure.firestore_store import (
    _plan_for_firestore,
    _plan_from_firestore,
)


def test_firestore_plan_round_trip_preserves_unsigned_64_bit_roll() -> None:
    original = {"move": "brief_close", "roll_uint64": 2**64 - 1}

    stored = _plan_for_firestore(original)
    restored = _plan_from_firestore(stored)

    assert stored is not None
    assert stored["roll_uint64"] == str(2**64 - 1)
    assert restored == original

import asyncio
from copy import deepcopy

from nora_quantica.application.audit import audit_session

from .test_conversation_service import make_service


def test_export_replays_without_inconsistencies(tmp_path) -> None:
    service, _, _, _ = make_service(tmp_path)
    state = asyncio.run(service.create_conversation())
    asyncio.run(service.send_message(state.conversation_id, "Primer mensaje"))
    asyncio.run(service.send_message(state.conversation_id, "Segundo mensaje"))
    exported = service.export_session(state.conversation_id)

    report = audit_session(exported)

    assert report.passed is True
    assert report.errors == []
    assert report.checks >= 6


def test_audit_detects_tampered_entropy_and_transition(tmp_path) -> None:
    service, _, _, _ = make_service(tmp_path)
    state = asyncio.run(service.create_conversation())
    asyncio.run(service.send_message(state.conversation_id, "Hola"))
    exported = service.export_session(state.conversation_id)
    tampered = deepcopy(exported)
    tampered["quantum_source"]["raw_bytes"][0] = 255
    tampered["turns"][0]["effective_deltas"]["candor"] = 99

    report = audit_session(tampered)

    assert report.passed is False
    assert any("hash SHA-256" in error for error in report.errors)
    assert any("deltas efectivos" in error for error in report.errors)


def test_audit_detects_tampered_prompt_profile(tmp_path) -> None:
    service, _, _, _ = make_service(tmp_path)
    state = asyncio.run(service.create_conversation())
    exported = service.export_session(state.conversation_id)
    exported["metadata"]["prompt_profile"]["base_prompt"] += " alterado"

    report = audit_session(exported)

    assert report.passed is False
    assert any("Perfil de prompt inválido" in error for error in report.errors)


def test_audit_detects_tampered_behavioral_plan(tmp_path) -> None:
    service, _, _, _ = make_service(tmp_path)
    state = asyncio.run(service.create_conversation())
    asyncio.run(service.send_message(state.conversation_id, "Hola"))
    exported = service.export_session(state.conversation_id)
    exported["turns"][0]["behavioral_plan"]["move"] = "brief_close"

    report = audit_session(exported)

    assert report.passed is False
    assert any("plan conductual" in error for error in report.errors)

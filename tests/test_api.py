from fastapi.testclient import TestClient

from nora_quantica.api import create_app
from nora_quantica.infrastructure.chat_models import ModelProviderError
from nora_quantica.settings import ModelSettings, Settings

from .test_conversation_service import make_service


def settings(database_path: str) -> Settings:
    model = ModelSettings("test", "test", "", "http://test", 0)
    return Settings(
        app_mode="demo",
        prompt_profile="baseline",
        database_path=database_path,
        anu_qrng_api_key=None,
        anu_qrng_url="http://test",
        evaluator=model,
        generator=model,
    )


def test_api_conversation_chat_lab_and_export(tmp_path) -> None:
    service, _, _, _ = make_service(tmp_path)
    app = create_app(service, settings(str(tmp_path / "api.db")))
    with TestClient(app) as client:
        homepage = client.get("/")
        assert homepage.status_code == 200
        assert "frame-ancestors 'none'" in homepage.headers["content-security-policy"]
        assert homepage.headers["x-content-type-options"] == "nosniff"
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/health").json() == {"status": "ok", "mode": "demo"}
        created = client.post("/api/conversations")
        assert created.status_code == 201
        assert created.json()["prompt_profile"] == "baseline"
        conversation_id = created.json()["conversation_id"]

        turn = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"message": "Hola"},
        )
        assert turn.status_code == 200
        assert turn.json()["response"] == "Respuesta natural"

        messages = client.get(f"/api/conversations/{conversation_id}/messages").json()
        assert len(messages["messages"]) == 2
        lab = client.get(f"/api/conversations/{conversation_id}/lab").json()
        assert lab["turn_number"] == 1
        assert lab["prompt_profile"]["name"] == "baseline"
        assert lab["last_turn"]["effective_deltas"]
        exported = client.get(f"/api/conversations/{conversation_id}/export")
        assert exported.status_code == 200
        assert "attachment" in exported.headers["content-disposition"]


def test_api_returns_404_and_rejects_empty_message(tmp_path) -> None:
    service, _, _, _ = make_service(tmp_path)
    app = create_app(service, settings(str(tmp_path / "api.db")))
    with TestClient(app) as client:
        assert client.get("/api/conversations/missing/messages").status_code == 404
        created = client.post("/api/conversations").json()
        response = client.post(
            f"/api/conversations/{created['conversation_id']}/messages",
            json={"message": ""},
        )
        assert response.status_code == 422


def test_api_maps_provider_failure_to_502_without_advancing_turn(tmp_path) -> None:
    service, store, _, _ = make_service(
        tmp_path,
        ModelProviderError("modelo temporalmente no disponible"),
    )
    app = create_app(service, settings(str(tmp_path / "api.db")))
    with TestClient(app) as client:
        conversation_id = client.post("/api/conversations").json()["conversation_id"]
        response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"message": "Hola"},
        )

    assert response.status_code == 502
    assert "temporalmente no disponible" in response.json()["detail"]
    restored = store.get_conversation(conversation_id)
    assert restored is not None and restored.turn_number == 0
    assert store.list_messages(conversation_id) == []

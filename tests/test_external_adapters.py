import asyncio
import json

import pytest

from nora_quantica.infrastructure.chat_models import (
    ModelProviderError,
    OpenAICompatibleChatModel,
)
from nora_quantica.infrastructure.entropy import AnuQuantumProvider, QuantumProviderError


class Headers(dict):
    pass


class Response:
    def __init__(self, payload: object, headers: dict[str, str] | None = None):
        self.payload = json.dumps(payload).encode()
        self.headers = Headers(headers or {})

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload


def test_anu_adapter_validates_uint8_and_records_request_id(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response(
            {"success": True, "data": [0, 64, 128, 192, 255]},
            {"x-amzn-requestid": "anu-request-1"},
        )

    monkeypatch.setattr("nora_quantica.infrastructure.entropy.urlopen", fake_urlopen)
    provider = AnuQuantumProvider("secret", timeout_seconds=3)

    sample = asyncio.run(provider.get_bytes(5))

    assert sample.raw_bytes == bytes([0, 64, 128, 192, 255])
    assert sample.metadata.provider == "anu_quantum_numbers"
    assert sample.metadata.response_id == "anu-request-1"
    assert captured["request"].get_header("X-api-key") == "secret"
    assert "length=5" in captured["request"].full_url
    assert captured["timeout"] == 3


def test_anu_adapter_rejects_out_of_range_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "nora_quantica.infrastructure.entropy.urlopen",
        lambda *_args, **_kwargs: Response({"success": True, "data": [256] * 5}),
    )
    with pytest.raises(QuantumProviderError, match="uint8"):
        asyncio.run(AnuQuantumProvider("secret").get_bytes(5))


def test_chat_adapter_sends_strict_schema_and_parses_content(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["authorization"] = request.get_header("Authorization")
        captured["url"] = request.full_url
        return Response({"choices": [{"message": {"content": '{"score":50}'}}]})

    monkeypatch.setattr("nora_quantica.infrastructure.chat_models.urlopen", fake_urlopen)
    model = OpenAICompatibleChatModel(
        provider="compatible",
        model="test-model",
        api_key="secret",
        base_url="https://models.example/v1/",
        temperature=0,
    )
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
        "additionalProperties": False,
    }

    output = asyncio.run(
        model.generate("system", [{"role": "user", "content": "hola"}], schema)
    )

    assert output == '{"score":50}'
    assert captured["url"] == "https://models.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert captured["payload"]["response_format"]["json_schema"]["strict"] is True
    assert captured["payload"]["messages"][0] == {"role": "system", "content": "system"}


def test_chat_adapter_rejects_malformed_provider_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "nora_quantica.infrastructure.chat_models.urlopen",
        lambda *_args, **_kwargs: Response({"unexpected": True}),
    )
    model = OpenAICompatibleChatModel(
        provider="compatible",
        model="test-model",
        api_key="secret",
        base_url="https://models.example/v1",
    )
    with pytest.raises(ModelProviderError, match="Respuesta invalida"):
        asyncio.run(model.generate("system", []))

import pytest

from explainrec.llm.backend import CliBackend, GeminiBackend, extract_json, get_backend
from explainrec.scenario import Modification


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_fenced_and_prosy():
    text = 'Here you go:\n```json\n{"summary": "x", "focal_users": [0]}\n```\nDone.'
    assert extract_json(text) == '{"summary": "x", "focal_users": [0]}'


def test_extract_json_missing_raises():
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("sorry, I cannot do that")


class FakeCli(CliBackend):
    """CliBackend with the subprocess call stubbed out."""

    def __init__(self, replies: list[str]):
        super().__init__()
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def text(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.replies.pop(0)


def test_cli_structured_parses_first_try():
    backend = FakeCli(['{"summary": "drop exploration", "remove_constraints": ["cold-item-exposure"]}'])
    mod = backend.structured("sys", "query", Modification)
    assert mod.remove_constraints == ["cold-item-exposure"]
    # schema is embedded in the system prompt
    assert "JSON schema" in backend.calls[0][0]


def test_cli_structured_retries_on_invalid_then_succeeds():
    backend = FakeCli([
        "I think you should remove the constraint.",   # invalid: no JSON
        '{"summary": "second try"}',
    ])
    mod = backend.structured("sys", "query", Modification)
    assert mod.summary == "second try"
    assert len(backend.calls) == 2
    assert "was not valid" in backend.calls[1][1]


def test_cli_structured_fails_after_retry():
    backend = FakeCli(["nope", "still nope"])
    with pytest.raises(ValueError):
        backend.structured("sys", "query", Modification)


def test_get_backend_unknown():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("magic")


class _FakeResponse:
    def __init__(self, text=None, parsed=None):
        self.text = text
        self.parsed = parsed


class _FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeGeminiClient:
    def __init__(self, response):
        self.models = _FakeModels(response)


def test_gemini_text_returns_response_text():
    client = _FakeGeminiClient(_FakeResponse(text="hello there"))
    backend = GeminiBackend(client=client)
    assert backend.text("sys", "hi") == "hello there"
    assert client.models.calls[0]["model"] == backend.model


def test_gemini_text_raises_on_empty_response():
    client = _FakeGeminiClient(_FakeResponse(text=""))
    backend = GeminiBackend(client=client)
    with pytest.raises(RuntimeError, match="no text"):
        backend.text("sys", "hi")


def test_gemini_structured_returns_parsed_model():
    mod = Modification(summary="drop exploration", remove_constraints=["cold-item-exposure"])
    client = _FakeGeminiClient(_FakeResponse(text="{...}", parsed=mod))
    backend = GeminiBackend(client=client)
    result = backend.structured("sys", "query", Modification)
    assert result.remove_constraints == ["cold-item-exposure"]


def test_gemini_structured_raises_when_unparsed():
    client = _FakeGeminiClient(_FakeResponse(text="not json", parsed=None))
    backend = GeminiBackend(client=client)
    with pytest.raises(RuntimeError, match="could not parse"):
        backend.structured("sys", "query", Modification)

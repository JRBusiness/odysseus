import asyncio
import json

from src import endpoint_resolver
from src import llm_core


class _FakeResp:
    status_code = 200

    async def aiter_lines(self):
        yield 'data: ' + json.dumps({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "ok"},
        })
        yield 'data: ' + json.dumps({"type": "message_stop"})

    async def aread(self):
        return b""


class _FakeStreamCtx:
    def __init__(self, capture, method, url, **kwargs):
        self.capture = capture
        self.capture.update(method=method, url=url, kwargs=kwargs)

    async def __aenter__(self):
        return _FakeResp()

    async def __aexit__(self, *args):
        return False


class _FakeClient:
    def __init__(self, capture):
        self.capture = capture

    def stream(self, method, url, **kwargs):
        return _FakeStreamCtx(self.capture, method, url, **kwargs)


def test_stream_anthropic_oauth_uses_claude_code_request_shape(monkeypatch):
    capture = {}
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: _FakeClient(capture))
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda url: False)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda *args, **kwargs: None)

    headers = endpoint_resolver.build_headers(
        "oauth:claude-oauth-token",
        "https://api.anthropic.com",
    )

    async def run():
        chunks = []
        async for chunk in llm_core.stream_llm(
            "https://api.anthropic.com/v1/messages",
            "claude-sonnet-4-5",
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Hello from OAuth"},
            ],
            headers=headers,
        ):
            chunks.append(chunk)
        return "".join(chunks)

    assert "ok" in asyncio.run(run())

    assert capture["url"] == "https://api.anthropic.com/v1/messages?beta=true"
    sent_headers = capture["kwargs"]["headers"]
    assert sent_headers["Authorization"] == "Bearer claude-oauth-token"
    assert "oauth-2025-04-20" in sent_headers["anthropic-beta"]
    assert "x-api-key" not in sent_headers

    payload = capture["kwargs"]["json"]
    assert payload["metadata"]["user_id"]
    assert payload["system"][0]["text"].startswith("x-anthropic-billing-header:")
    assert payload["system"][1]["text"].startswith("You are Claude Code")
    assert payload["system"][2]["text"] == "System prompt"


def test_anthropic_oauth_url_appends_beta_to_existing_query():
    assert (
        llm_core._anthropic_oauth_url("https://api.anthropic.com/v1/messages?foo=bar")
        == "https://api.anthropic.com/v1/messages?foo=bar&beta=true"
    )

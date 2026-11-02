from types import SimpleNamespace

import pytest

from ledgerlens.config import get_settings
from ledgerlens.llm import provider
from ledgerlens.llm.cache import CacheMissError, ResponseCache, cache_key
from ledgerlens.llm.models import MODELS, get_model


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("LLM_MODE", "live")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def _fake_raw(content="391", prompt=20, completion=4, total=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total or prompt + completion,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_cache_key_is_order_insensitive_and_request_sensitive():
    a = cache_key("llm", {"model": "m", "temperature": 0, "messages": [{"role": "user"}]})
    b = cache_key("llm", {"messages": [{"role": "user"}], "temperature": 0, "model": "m"})
    c = cache_key("llm", {"model": "m", "temperature": 0.1, "messages": [{"role": "user"}]})
    assert a == b != c
    assert cache_key("tool", {"x": 1}) != cache_key("llm", {"x": 1})


def test_cache_reads_replay_dirs(tmp_path):
    replay = ResponseCache(tmp_path / "replay")
    replay.put("llm", "abcd", {"content": "x"})
    cache = ResponseCache(tmp_path / "main", replay_dirs=[tmp_path / "replay"])
    assert cache.get("llm", "abcd") == {"content": "x"}
    assert cache.get("llm", "ffff") is None


def test_cost_uses_list_prices():
    spec = get_model("gemini-3.8-flash")
    # 1M input at $0.75 + 1M output at $3.75
    assert spec.cost(1_000_000, 1_000_000) == pytest.approx(4.50)
    assert get_model("ollama-qwen3-8b").cost(10**6, 10**6) == 0.0
    with pytest.raises(KeyError):
        get_model("gpt-unknown")


def test_judge_family_differs_from_answer_models():
    judge = MODELS["groq-gpt-oss-120b"].family
    assert all(MODELS[m].family != judge for m in ("gemini-3.8-flash", "groq-qwen3.8-27b"))


def test_output_tokens_include_hidden_thinking():
    # Gemini's OpenAI endpoint: prompt 3, completion 1, total 61 -> 58 thinking+output tokens.
    assert (
        provider._output_tokens(
            SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=61)
        )
        == 58
    )
    assert (
        provider._output_tokens(
            SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)
        )
        == 5
    )


def test_complete_caches_and_tracks_usage(isolated, monkeypatch):
    calls = []

    def fake_call(spec, request):
        calls.append(request)
        return _fake_raw(content="<think>hmm</think>391", prompt=20, completion=4), 123.4

    monkeypatch.setattr(provider, "_call_provider", fake_call)
    msgs = [{"role": "user", "content": "17*23?"}]
    with provider.track_usage() as outer:
        first = provider.complete("groq-qwen3.8-27b", msgs)
        with provider.track_usage() as inner:
            second = provider.complete("groq-qwen3.8-27b", msgs)
    assert len(calls) == 1  # second call served from disk
    assert first.content == second.content == "391"  # think block stripped
    assert not first.cached and second.cached
    assert second.latency_ms == 123.4  # replay keeps recorded latency
    assert outer.total.llm_calls == 2 and inner.total.llm_calls == 1
    assert outer.total.cost_usd == pytest.approx(2 * (20 * 0.80 + 4 * 4.00) / 1e6)


def test_replay_mode_raises_on_miss(isolated, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "replay")
    get_settings.cache_clear()
    monkeypatch.setattr(provider, "_call_provider", lambda *a: pytest.fail("network in replay"))
    with pytest.raises(CacheMissError):
        provider.complete("gemini-3.8-flash", [{"role": "user", "content": "hi"}])


def test_tool_calls_round_trip(isolated, monkeypatch):
    tc = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="calculate", arguments='{"expression": "1+1"}')
    )
    monkeypatch.setattr(
        provider, "_call_provider", lambda s, r: (_fake_raw(content=None, tool_calls=[tc]), 5.0)
    )
    tools = [{"type": "function", "function": {"name": "calculate", "parameters": {}}}]
    first = provider.complete("gemini-3.8-flash", [{"role": "user", "content": "x"}], tools=tools)
    second = provider.complete("gemini-3.8-flash", [{"role": "user", "content": "x"}], tools=tools)
    assert first.tool_calls[0].name == second.tool_calls[0].name == "calculate"
    assert second.tool_calls[0].arguments == '{"expression": "1+1"}'


@pytest.mark.parametrize(
    ("name", "retry"),
    [
        ("RateLimitError", True),
        ("ServiceUnavailableError", True),
        ("AuthenticationError", False),
        ("BadRequestError", False),
        ("ValueError", False),
    ],
)
def test_retry_classification(name, retry):
    exc_type = type(name, (Exception,), {})
    assert provider._is_retryable(exc_type()) is retry


def test_retry_skips_auth_errors_even_if_subclass_of_api_error():
    api_error = type("APIError", (Exception,), {})
    auth = type("AuthenticationError", (api_error,), {})
    assert provider._is_retryable(auth()) is False

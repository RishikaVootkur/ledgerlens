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


def test_retry_classification_with_real_litellm_classes():
    import litellm.exceptions as ex

    def make(cls):
        return cls.__new__(cls)  # constructors differ; only the type matters here

    for cls in (
        ex.RateLimitError,
        ex.ServiceUnavailableError,
        ex.InternalServerError,
        ex.APIConnectionError,
        ex.Timeout,
        ex.BadGatewayError,
    ):
        assert provider._is_retryable(make(cls)), cls.__name__
    for cls in (
        ex.AuthenticationError,
        ex.PermissionDeniedError,
        ex.BadRequestError,
        ex.NotFoundError,
        ex.UnprocessableEntityError,
        ex.ContextWindowExceededError,
        ex.APIResponseValidationError,
        ex.JSONSchemaValidationError,
        ex.APIError,
    ):
        assert not provider._is_retryable(make(cls)), cls.__name__
    assert not provider._is_retryable(provider.QuotaExhaustedError("day"))


def test_retry_after_header_is_honored():
    exc = Exception("slow down")
    exc.headers = {"retry-after": "7"}
    assert provider._retry_after(exc) == 7.0
    assert provider._retry_after(Exception("no headers")) is None


def test_daily_quota_stops_immediately(isolated, monkeypatch):
    import litellm
    import litellm.exceptions as ex

    def boom(**_):
        raise ex.RateLimitError("Rate limit reached: tokens per day (TPD) exceeded", "groq", "m")

    monkeypatch.setattr(litellm, "completion", boom)
    monkeypatch.setattr(provider._pacer, "wait", lambda *a, **k: None)
    with pytest.raises(provider.QuotaExhaustedError):
        provider.complete("groq-gpt-oss-120b", [{"role": "user", "content": "x"}])


def test_pacer_respects_token_budget(monkeypatch):
    sleeps = []
    clock = [1000.0]
    monkeypatch.setattr(provider.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(provider.time, "sleep", lambda s: sleeps.append(s))
    pacer = provider._Pacer()
    spec = get_model("groq-gpt-oss-120b")  # 30 rpm, 8000 tpm
    pacer.wait(spec, 5000)
    pacer.wait(spec, 5000)  # would exceed 8000 in the window -> wait until the first ages out
    assert sleeps[-1] == pytest.approx(60.0)
    pacer2 = provider._Pacer()
    sleeps.clear()
    pacer2.wait(spec, 100)
    pacer2.wait(spec, 100)  # only the request spacing applies: 60/30 = 2 s
    assert sleeps == [pytest.approx(2.0)]


def test_concurrent_cache_writes_do_not_collide(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    cache = ResponseCache(tmp_path)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: cache.put("llm", "samekey", {"i": i}), range(200)))
    assert cache.get("llm", "samekey") is not None
    assert not list(tmp_path.rglob("*.tmp"))


def test_empty_answers_are_not_cached(isolated, monkeypatch):
    calls = []

    def fake(spec, request):
        calls.append(1)
        return _fake_raw(content="", completion=0), 1.0

    monkeypatch.setattr(provider, "_call_provider", fake)
    provider.complete("gemini-3.8-flash", [{"role": "user", "content": "x"}])
    provider.complete("gemini-3.8-flash", [{"role": "user", "content": "x"}])
    assert len(calls) == 2


def test_replay_reads_only_committed_cache(isolated, monkeypatch):
    monkeypatch.setattr(provider, "_call_provider", lambda s, r: (_fake_raw(content="live"), 9.0))
    msgs = [{"role": "user", "content": "q"}]
    provider.complete("gemini-3.8-flash", msgs)  # recorded into the local .cache only
    monkeypatch.setenv("LLM_MODE", "replay")
    get_settings.cache_clear()
    with pytest.raises(CacheMissError):
        provider.complete("gemini-3.8-flash", msgs)
    # Promote the entry into the committed cache: now replay serves it.
    local = next((isolated / "cache" / "llm").rglob("*.json"))
    dest = isolated / "eval" / "cache" / "llm" / local.parent.name / local.name
    dest.parent.mkdir(parents=True)
    dest.write_text(local.read_text())
    assert provider.complete("gemini-3.8-flash", msgs).content == "live"


def test_temperature_int_and_float_share_a_key(isolated, monkeypatch):
    calls = []
    monkeypatch.setattr(
        provider, "_call_provider", lambda s, r: (calls.append(1), (_fake_raw(), 1.0))[1]
    )
    provider.complete("gemini-3.8-flash", [{"role": "user", "content": "t"}], temperature=0)
    provider.complete("gemini-3.8-flash", [{"role": "user", "content": "t"}], temperature=0.0)
    assert len(calls) == 1


def test_unclosed_think_block_is_stripped():
    assert provider._THINK_RE.sub("", "<think>never closed").strip() == ""

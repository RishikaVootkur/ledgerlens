import asyncio
import json
import re
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ledgerlens.agent import graph
from ledgerlens.api import app as api_module
from ledgerlens.api import facts
from ledgerlens.api.app import ChatRequest, chat_events, create_app
from ledgerlens.config import get_settings
from ledgerlens.llm.provider import QuotaExhaustedError
from ledgerlens.retrieval import retriever
from ledgerlens.retrieval.store import ChunkStore
from ledgerlens.schemas import AgentAnswer, Chunk, Citation, Usage
from ledgerlens.universe import UNIVERSE

FIXTURES = Path(__file__).parents[1] / "eval_datasets" / "fixtures"
CHUNK_ID = "AAPL-10K-2025FY:section:00042"
KO_FACT = "xbrl:KO:NetIncomeLoss:2026-07-04_2026-10-02:USD"

ANSWER = AgentAnswer(
    answer="Apple's total net sales were $416,161 million in fiscal 2025 [1].",
    answer_value=416_161e6,
    answer_unit="USD",
    citations=[
        Citation(
            chunk_id=CHUNK_ID,
            filing_id="AAPL-10K-2025FY",
            section="Item 7",
            page=21,
            quote="Total net sales were $416,161 million",
        )
    ],
    retrieved_chunk_ids=[CHUNK_ID],
    steps=["plan", "search_filings", "answer", "verify", "self_check"],
    usage=Usage(prompt_tokens=900, completion_tokens=80, llm_calls=2, cost_usd=0.001),
)


def make_chunk(chunk_id: str = CHUNK_ID) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        filing_id="AAPL-10K-2025FY",
        ticker="AAPL",
        company="Apple Inc.",
        form="10-K",
        fiscal_year=2025,
        fiscal_period="FY",
        strategy="section",
        section="Item 7",
        page=21,
        text="Total net sales were $416,161 million in 2025.",
    )


@pytest.fixture
def env(monkeypatch, tmp_path):
    data, ev = tmp_path / "data", tmp_path / "eval"
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("EVAL_DIR", str(ev))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    get_settings.cache_clear()
    retriever.get_store.cache_clear()
    monkeypatch.setattr(api_module, "_tickers_cache", (0.0, frozenset()))
    yield tmp_path
    get_settings.cache_clear()
    retriever.get_store.cache_clear()


@pytest.fixture
def store(env):
    s = ChunkStore(env / "data" / "index" / "chunks.sqlite")
    s.upsert([make_chunk()])
    return s


@pytest.fixture
def client(env):
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def calls(monkeypatch):
    """Replace the agent with a scripted one; record how it was called."""
    seen: list[dict] = []

    def fake_run_agent(question, config, filters, on_step):
        seen.append(
            {
                "question": question,
                "model": config.model,
                "filters": filters,
                "thread": threading.current_thread().name,
            }
        )
        on_step("plan", "round 1")
        on_step("search_filings", '{"query": "Apple net sales"}')
        on_step("answer", "writing a cited answer")
        return ANSWER

    monkeypatch.setattr(graph, "run_agent", fake_run_agent)
    return seen


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        name, data = None, []
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data.append(line[len("data: ") :])
        if name:
            events.append((name, json.loads("\n".join(data))))
    return events


def post_chat(client, payload):
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    return parse_sse(resp.text)


# ------------------------------------------------------------------ chat


def test_chat_streams_steps_tokens_answer_done(client, calls):
    events = post_chat(client, {"question": "  What were Apple's net sales in 2025? "})
    names = [n for n, _ in events]
    assert names[:3] == ["step", "step", "step"]
    assert [d["name"] for _, d in events[:3]] == ["plan", "search_filings", "answer"]
    assert events[1][1]["detail"] == '{"query": "Apple net sales"}'
    tokens = [d["text"] for n, d in events if n == "token"]
    assert len(tokens) > 1
    assert "".join(tokens) == ANSWER.answer
    assert names[-2:] == ["answer", "done"]
    assert set(names[3:-2]) == {"token"}
    assert AgentAnswer.model_validate(events[-2][1]) == ANSWER
    assert events[-1][1] == {}
    assert calls[0]["question"] == "What were Apple's net sales in 2025?"
    assert calls[0]["model"] == "groq-qwen3.8-27b"
    assert calls[0]["thread"].startswith("agent")


def test_chat_passes_filters_and_model(client, calls):
    post_chat(
        client,
        {
            "question": "KO revenue?",
            "filters": {"ticker": "ko", "fiscal_year": 2025, "form": "10-K"},
            "model": "gemini-3.8-flash",
        },
    )
    f = calls[0]["filters"]
    assert (f.ticker, f.fiscal_year, f.form) == ("KO", 2025, "10-K")
    assert calls[0]["model"] == "gemini-3.8-flash"


def test_chat_agent_exception_becomes_error_event(client, monkeypatch):
    def boom(question, config, filters, on_step):
        on_step("plan", "round 1")
        raise RuntimeError("internal detail that should stay in the log")

    monkeypatch.setattr(graph, "run_agent", boom)
    events = post_chat(client, {"question": "anything"})
    assert [n for n, _ in events] == ["step", "error", "done"]
    message = events[1][1]["message"]
    assert "RuntimeError" in message
    assert "internal detail" not in message


def test_chat_quota_error_is_explained(client, monkeypatch):
    def quota(question, config, filters, on_step):
        raise QuotaExhaustedError("groq-qwen3.8-27b: daily free-tier quota exhausted")

    monkeypatch.setattr(graph, "run_agent", quota)
    events = post_chat(client, {"question": "anything"})
    assert [n for n, _ in events] == ["error", "done"]
    assert "quota" in events[0][1]["message"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 1001},
        {"question": "q", "model": "gpt-5"},
        {"question": "q", "filters": {"ticker": "ZZZZ"}},
        {"question": "q", "filters": {"fiscal_year": 1800}},
        {"question": "q", "filters": {"form": "8-K"}},
        {"question": "q", "filters": {"sector": "Energy"}},
        {"question": "q", "extra": True},
    ],
)
def test_chat_rejects_invalid_requests(client, calls, payload):
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 422
    assert calls == []


def test_blank_optional_filters_are_ignored():
    req = ChatRequest.model_validate({"question": "q", "filters": {"ticker": "", "form": ""}})
    assert req.filters.ticker is None and req.filters.form is None


async def test_disconnect_stops_the_agent(monkeypatch):
    """Closing the stream (what sse-starlette does on disconnect) makes the agent's next
    step raise, so an abandoned question stops spending LLM calls."""
    from concurrent.futures import ThreadPoolExecutor

    release, finished = threading.Event(), threading.Event()
    outcome: list[str] = []

    def slow(question, config, filters, on_step):
        on_step("plan", "round 1")
        release.wait(5)
        try:
            on_step("answer", "never sent")
            outcome.append("continued")
        except api_module.ChatCancelledError:
            outcome.append("cancelled")
            raise
        finally:
            finished.set()
        return ANSWER

    monkeypatch.setattr(graph, "run_agent", slow)
    executor = ThreadPoolExecutor(max_workers=1)
    stream = chat_events(ChatRequest(question="q"), executor, asyncio.Semaphore(1), "test")
    first = await anext(stream)
    assert first.event == "step"
    await stream.aclose()
    release.set()
    await asyncio.to_thread(finished.wait, 5)
    executor.shutdown(wait=True)
    assert outcome == ["cancelled"]


def test_token_pieces_round_trip():
    text = "Net sales were  $416,161 million\nin 2025 [1]."
    pieces = api_module._token_pieces(text)
    assert "".join(pieces) == text
    assert len(pieces) > 1
    assert api_module._token_pieces("") == []


# ------------------------------------------------------------------ chunks


def test_chunk_from_store(client, store):
    resp = client.get(f"/api/chunks/{CHUNK_ID.replace(':', '%3A')}")
    assert resp.status_code == 200
    assert Chunk.model_validate(resp.json()) == make_chunk()


def test_unknown_chunk_is_404(client, store):
    assert client.get("/api/chunks/AAPL-10K-2025FY%3Asection%3A99999").status_code == 404


def test_missing_store_is_404_and_not_created(client, env):
    assert client.get(f"/api/chunks/{CHUNK_ID}").status_code == 404
    assert not (env / "data" / "index" / "chunks.sqlite").exists()


@pytest.fixture
def ko_facts(monkeypatch):
    data = json.loads((FIXTURES / "companyfacts_ko.json").read_text())
    for taxonomy in data["facts"].values():
        for concept in taxonomy.values():
            for rows in concept["units"].values():
                for row in rows:
                    row.setdefault("filed", "2026-10-30")
    monkeypatch.setattr(facts, "_companyfacts", lambda ticker: data)


def test_xbrl_fact_chunk(client, ko_facts):
    resp = client.get(f"/api/chunks/{KO_FACT.replace(':', '%3A')}")
    assert resp.status_code == 200
    chunk = Chunk.model_validate(resp.json())
    assert chunk.chunk_id == KO_FACT
    assert (chunk.ticker, chunk.form, chunk.fiscal_year, chunk.fiscal_period) == (
        "KO",
        "10-Q",
        2026,
        "Q3",
    )
    assert chunk.filing_id == "KO-10Q-2026Q3"
    assert chunk.company == "The Coca-Cola Company"
    assert "3,000 USD" in chunk.text
    assert "2026-07-04 to 2026-10-02 (Q)" in chunk.text
    assert "us-gaap:NetIncomeLoss" in chunk.text


def test_xbrl_unit_with_slash(client, monkeypatch):
    data = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "label": "EPS diluted",
                    "units": {
                        "USD/shares": [
                            {"start": "2025-01-01", "end": "2025-12-31", "val": 2.81,
                             "accn": "a", "fy": 2025, "fp": "FY", "form": "10-K",
                             "filed": "2026-02-20"}
                        ]
                    },
                }
            }
        }
    }  # fmt: skip
    monkeypatch.setattr(facts, "_companyfacts", lambda ticker: data)
    fid = "xbrl:KO:EarningsPerShareDiluted:2025-01-01_2025-12-31:USD/shares"
    resp = client.get("/api/chunks/" + fid.replace(":", "%3A").replace("/", "%2F"))
    assert resp.status_code == 200
    assert "2.81 USD/shares" in resp.json()["text"]


@pytest.mark.parametrize(
    "fact_id",
    [
        "xbrl:KO:NetIncomeLoss:2020-01-01_2020-12-31:USD",  # no such period
        "xbrl:KO:NoSuchConcept:2026-07-04_2026-10-02:USD",
        "xbrl:ZZZZ:NetIncomeLoss:2026-07-04_2026-10-02:USD",  # not in the universe
        "xbrl:KO:NetIncomeLoss:not-a-date:USD",
        "xbrl:KO:NetIncomeLoss",
    ],
)
def test_bad_xbrl_ids_are_404(client, ko_facts, fact_id):
    assert client.get("/api/chunks/" + fact_id.replace(":", "%3A")).status_code == 404


def test_xbrl_lookup_failure_still_describes_the_fact(client, monkeypatch):
    def offline(ticker):
        raise OSError("network down")

    monkeypatch.setattr(facts, "_companyfacts", offline)
    resp = client.get("/api/chunks/" + KO_FACT)
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunk_id"] == KO_FACT
    assert "could not be loaded" in body["text"]


# ------------------------------------------------------------------ models, companies, health


def test_models(client):
    body = client.get("/api/models").json()
    ids = [m["id"] for m in body["models"]]
    assert body["default"] in ids
    assert "groq-qwen3.8-27b" in ids
    assert all(set(m) == {"id", "label", "provider"} for m in body["models"])


def test_companies_are_the_indexed_ones(client, store):
    body = client.get("/api/companies").json()
    assert body == {
        "companies": [{"ticker": "AAPL", "name": "Apple Inc.", "sector": "Information Technology"}]
    }


def test_companies_fall_back_to_universe(client):
    assert len(client.get("/api/companies").json()["companies"]) == len(UNIVERSE)


def test_health(client, store, monkeypatch):
    async def fake_http(url):
        return {"ok": "6333" in url, "detail": "HTTP 200" if "6333" in url else "ConnectError"}

    monkeypatch.setattr(api_module, "_check_http", fake_http)
    body = client.get("/api/health").json()
    assert body["checks"]["store"] == {"ok": True, "detail": "1 chunks"}
    assert body["checks"]["qdrant"]["ok"] is True
    assert body["checks"]["ollama"]["ok"] is False
    assert body["status"] == "degraded"


def test_cors_allows_vite_dev_origin(client):
    resp = client.options(
        "/api/models",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = client.get("/api/models", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_request_id_header(client):
    resp = client.get("/api/models", headers={"x-request-id": "abc-123.x_y"})
    assert resp.headers["x-request-id"] == "abc-123.x_y"


@pytest.mark.parametrize("raw", ["a" * 65, "has space", "line\\nbreak", "<script>", ""])
def test_unsafe_request_ids_are_replaced(client, raw):
    returned = client.get("/api/models", headers={"x-request-id": raw}).headers["x-request-id"]
    assert returned != raw
    assert re.fullmatch(r"[0-9a-f]{12}", returned)


def test_access_log_times_the_whole_stream(client, monkeypatch, calls):
    """The chat is logged once, after the last body chunk, so `ms` covers the stream."""
    logged: list[dict] = []
    monkeypatch.setattr(api_module, "TOKEN_DELAY_S", 0.05)

    class Spy:
        def info(self, event, **kw):
            if event == "http.request":
                logged.append(kw)

        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(api_module, "log", Spy())
    post_chat(client, {"question": "q"})
    (entry,) = [e for e in logged if e["path"] == "/api/chat"]
    assert entry["status"] == 200
    pieces = len(api_module._token_pieces(ANSWER.answer))
    assert entry["ms"] >= pieces * 50 * 0.8


# ------------------------------------------------------------------ eval data


@pytest.fixture
def eval_files(env):
    ev = env / "eval"
    (ev / "reports").mkdir(parents=True)
    (ev / "runs").mkdir()
    (ev / "reports" / "dashboard.json").write_text('{"generated_at": "2026-11-01"}')
    (ev / "runs" / "2026-11-01_default.json").write_text('{"run_id": "2026-11-01_default"}')
    (ev / "secret.json").write_text('{"secret": true}')
    (env / "outside.json").write_text('{"secret": true}')
    return ev


def test_data_missing_is_404(client):
    assert client.get("/api/data/dashboard.json").status_code == 404
    assert client.get("/api/data/runs/nope.json").status_code == 404


def test_data_served(client, eval_files):
    resp = client.get("/api/data/dashboard.json")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == {"generated_at": "2026-11-01"}
    run = client.get("/api/data/runs/2026-11-01_default.json")
    assert run.json() == {"run_id": "2026-11-01_default"}


@pytest.mark.parametrize(
    "path",
    [
        "/api/data/runs/..%2Fsecret.json",
        "/api/data/runs/..%2F..%2Foutside.json",
        "/api/data/runs/%2E%2E%2Fsecret.json",
        "/api/data/runs/../secret.json",
        "/api/data/runs/..json",
        "/api/data/runs/a..b.json",
        "/api/data/runs/.hidden.json",
        "/api/data/runs/%2Fetc%2Fpasswd.json",
    ],
)
def test_data_rejects_traversal(client, eval_files, path):
    resp = client.get(path)
    assert resp.status_code == 404
    assert "secret" not in resp.text

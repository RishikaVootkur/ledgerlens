"""Chat behaviour through a real uvicorn server and real HTTP connections."""

import asyncio
import json
import threading
import time

import httpx
import pytest
import uvicorn

from ledgerlens.agent import graph
from ledgerlens.api import app as api_module
from ledgerlens.api.app import create_app
from ledgerlens.config import get_settings
from ledgerlens.retrieval import retriever
from ledgerlens.schemas import AgentAnswer

ANSWER = AgentAnswer(answer="Done [1].", refused=False)


@pytest.fixture
def live_server(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MAX_CONCURRENT_CHATS", "2")
    get_settings.cache_clear()
    retriever.get_store.cache_clear()
    server = uvicorn.Server(
        uvicorn.Config(create_app(), host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(10)
    get_settings.cache_clear()
    retriever.get_store.cache_clear()


async def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.02)


async def test_real_disconnect_stops_the_agent(live_server, monkeypatch):
    release, finished = threading.Event(), threading.Event()
    outcome: list[str] = []

    def slow(question, config, filters, on_step):
        on_step("plan", "round 1")
        release.wait(10)
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
    async with (
        httpx.AsyncClient(base_url=live_server, timeout=10) as client,
        client.stream("POST", "/api/chat", json={"question": "q"}) as resp,
    ):
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                break  # leave after the first event; closing the stream disconnects
    # Give the server a moment to notice the disconnect, then let the agent continue.
    await asyncio.sleep(0.5)
    release.set()
    await asyncio.to_thread(finished.wait, 10)
    assert outcome == ["cancelled"]


async def test_concurrent_chats_are_capped(live_server, monkeypatch):
    release = threading.Event()
    lock = threading.Lock()
    active, peak = [0], [0]

    def blocking(question, config, filters, on_step):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        try:
            on_step("plan", question)
            release.wait(10)
            return ANSWER
        finally:
            with lock:
                active[0] -= 1

    monkeypatch.setattr(graph, "run_agent", blocking)
    first_events: dict[int, str] = {}

    async def chat(client, i):
        names = []
        async with client.stream("POST", "/api/chat", json={"question": f"q{i}"}) as resp:
            async for line in resp.aiter_lines():
                # Label each event by its step name, or by its type for non-step events.
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1])
                    names.append(data["name"] if event == "step" else event)
                    first_events.setdefault(i, names[0])
        return names

    async with httpx.AsyncClient(base_url=live_server, timeout=20) as client:
        tasks = [asyncio.create_task(chat(client, i)) for i in (0, 1)]
        await wait_for(lambda: active[0] == 2)
        third = asyncio.create_task(chat(client, 2))
        await wait_for(lambda: 2 in first_events)
        assert first_events[2] == "queued"
        await asyncio.sleep(0.3)
        assert active[0] == 2  # the third run has not started
        release.set()
        results = await asyncio.gather(*tasks, third)

    assert peak[0] == 2
    for names in results:
        assert names[-2:] == ["answer", "done"]
    assert results[2][:2] == ["queued", "plan"]

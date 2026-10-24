import asyncio
import json
from pathlib import Path

import pytest

from clausetrail.config import PROJECT_ROOT, Settings

SAMPLES = PROJECT_ROOT / "data" / "samples"
WHOLE_LIFE = SAMPLES / "chinalife-whole-life-hengyu-yisheng.pdf"


def make_settings(tmp: Path, **overrides) -> Settings:
    base = dict(
        _env_file=None,
        embedder_backend="fake",
        reranker_backend="fake",
        llm_backend="fake",
        bm25_backend="milvus",
        redis_url="",
        vector_store_uri=str(tmp / "milvus.db"),
        sqlite_path=tmp / "app.sqlite3",
        layout_analysis=False,
        refusal_threshold=0.3,
        allowed_hosts=["testserver", "localhost", "127.0.0.1"],
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(scope="module")
def settings(tmp_path_factory) -> Settings:
    return make_settings(tmp_path_factory.mktemp("ct"))


@pytest.fixture(scope="module")
def services(settings):
    from clausetrail.services import build_services

    svc = build_services(settings)
    yield svc
    svc.close()


@pytest.fixture(scope="module")
def ingested(services):
    from clausetrail.ingest.pipeline import ingest_file

    return asyncio.run(ingest_file(services, WHOLE_LIFE))


@pytest.fixture(scope="module")
def client(services, ingested):
    from fastapi.testclient import TestClient

    from clausetrail.api.app import create_app

    with TestClient(create_app(services)) as c:
        yield c


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        name, data = None, []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if name and data:
            events.append((name, json.loads("\n".join(data))))
    return events


def sse(client, message: str, session_id: str | None = None) -> list[tuple[str, dict]]:
    payload = {"message": message, **({"session_id": session_id} if session_id else {})}
    with client.stream("POST", "/api/chat", json=payload) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    return parse_sse(body)

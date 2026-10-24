import asyncio

from clausetrail.chat import ChatService
from clausetrail.retrieval.pipeline import Retriever
from clausetrail.services import build_services
from tests.conftest import WHOLE_LIFE, make_settings, sse


def test_ingest_report(ingested):
    assert ingested.file_name == WHOLE_LIFE.name
    assert ingested.pages == 8
    assert ingested.chunks > 20
    assert ingested.kb_version == 1


def test_hybrid_retrieval_finds_cooling_off_clause(services, ingested):
    result = asyncio.run(Retriever(services).retrieve("犹豫期是多久"))
    assert result.dense_hits > 0 and result.bm25_hits > 0
    assert all(c.rerank_score is not None for c in result.candidates)
    assert len(result.evidence) <= services.settings.context_top_k
    top = result.evidence[0].chunk
    assert "犹豫期" in top.text or "犹豫期" in top.section_label
    assert top.page_start >= 1 and top.section_path


def test_chat_stream_answers_with_citations(client):
    events = sse(client, "犹豫期是多久")
    names = [n for n, _ in events]
    assert names[0] == "meta" and names[1] == "retrieval" and names[-1] == "done"
    assert "delta" in names
    meta, retrieval, done = events[0][1], events[1][1], events[-1][1]
    assert meta["topic"] == "new" and meta["cache_hit"] is False
    assert retrieval["citations"][0]["n"] == 1 and retrieval["bm25_backend"] == "milvus_bm25"
    assert done["status"] == "answered" and done["used_citations"]
    answer = "".join(d["text"] for n, d in events if n == "delta")
    assert "[1]" in answer or "[2]" in answer


def test_second_identical_question_hits_cache(client):
    # 自己先问一次，不依赖前一个用例已经把答案写进缓存
    sse(client, "犹豫期是多久")
    events = sse(client, "犹豫期是多久？")
    assert events[0][1]["cache_hit"] is True
    assert events[1][1]["cached"] is True
    assert events[-1][1]["cache_hit"] is True and events[-1][1]["status"] == "answered"


def test_follow_up_is_rewritten_with_context(client):
    first = sse(client, "这款终身寿险的犹豫期有多长")
    session_id = first[-1][1]["session_id"]
    follow = sse(client, "那退保呢", session_id=session_id)
    meta = follow[0][1]
    assert meta["topic"] == "continue" and meta["topic_reason"] == "ellipsis"
    assert meta["rewritten_query"] and "犹豫期" in meta["rewritten_query"]
    detail = client.get(f"/api/sessions/{session_id}").json()
    assert len(detail["turns"]) == 2
    assert detail["turns"][1]["standalone_query"] == meta["rewritten_query"]


def test_off_topic_question_is_refused_without_generation(client):
    events = sse(client, "火锅底料的配方")
    names = [n for n, _ in events]
    assert "refusal" in names and "delta" not in names
    refusal = dict(events)["refusal"]
    assert refusal["reason"] in ("low_score", "no_hit")
    assert events[-1][1]["status"] == "refused"


def test_answer_without_citations_is_retracted(client, services, monkeypatch):
    async def no_citations(*args, **kwargs):
        yield "现有条款中没有找到可以回答该问题的内容。"

    monkeypatch.setattr(services.llm, "stream", no_citations)
    events = sse(client, "犹豫期是多久呀")
    assert dict(events)["refusal"]["reason"] == "unsupported"
    done = events[-1][1]
    assert done["status"] == "refused"
    turn = client.get(f"/api/sessions/{done['session_id']}").json()["turns"][-1]
    assert turn["answer"] == "" and turn["status"] == "refused"


def test_disconnect_after_refusal_keeps_refused_status(services, ingested):
    async def run():
        gen = ChatService(services).stream("火锅底料的配方")
        session_id = None
        async for name, data in gen:
            if name == "meta":
                session_id = data["session_id"]
            if name == "refusal":
                break
        # 模拟前端在 refusal 和 done 之间断开
        await gen.aclose()
        return session_id

    session_id = asyncio.run(run())
    assert services.db.turns(session_id)[-1]["status"] == "refused"


def test_documents_chunks_and_file_endpoints(client, ingested):
    docs = client.get("/api/documents").json()
    assert docs["kb_version"] == 1 and docs["documents"][0]["doc_id"] == ingested.doc_id
    assert "path" not in docs["documents"][0]
    citation = sse(client, "责任免除有哪些")[1][1]["citations"][0]
    chunk = client.get(f"/api/documents/{citation['doc_id']}/chunks/{citation['chunk_id']}").json()
    assert chunk["text"] == citation["snippet"]
    pdf = client.get(f"/api/documents/{ingested.doc_id}/file")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert client.get("/api/documents/0000000000000000/file").status_code == 404


def test_health_and_validation(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    comps = health["components"]
    assert comps["vector_store"]["index"] == "HNSW"
    assert comps["cache"]["status"] == "degraded"
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x", "session_id": "../etc"}).status_code == 422
    assert client.get("/api/sessions/not-a-session").status_code == 422
    # DNS rebinding：Host 不在白名单里的请求直接拒绝
    assert client.get("/api/health", headers={"host": "evil.example"}).status_code == 400
    # 白名单只比主机名，带端口的 Host（经 Vite 代理时是 localhost:5173）照常通过
    assert client.get("/api/health", headers={"host": "localhost:5173"}).status_code == 200


def test_missing_aliyun_keys_report_unavailable(tmp_path):
    services = build_services(
        make_settings(
            tmp_path,
            embedder_backend="aliyun_pai_eas",
            reranker_backend="aliyun_opensearch",
            llm_backend="aliyun_bailian",
        )
    )
    try:
        assert set(services.errors) == {"embedder", "reranker", "llm"}

        async def run():
            return [e async for e in ChatService(services).stream("等待期多久")]

        events = asyncio.run(run())
        assert events == [("error", {"code": "model_unavailable", "message": events[0][1]["message"]})]
    finally:
        services.close()

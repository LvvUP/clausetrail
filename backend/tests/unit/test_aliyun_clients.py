import asyncio
import json

import httpx
import numpy as np
import pytest

from clausetrail.models.embedder import AliyunPaiEasEmbedder
from clausetrail.models.llm import BailianLLM
from clausetrail.models.reranker import AliyunOpenSearchReranker
from tests.conftest import make_settings


@pytest.fixture
def cfg(tmp_path):
    return make_settings(
        tmp_path,
        aliyun_opensearch_host="https://opensearch.example.com",
        aliyun_opensearch_api_key="OS-test",
        aliyun_pai_eas_url="https://eas.example.com/api/predict/bge-m3",
        aliyun_pai_eas_token="eas-test",
        aliyun_bailian_api_key="sk-test",
        aliyun_bailian_base_url="https://bailian.example.com/compatible-mode/v1",
    )


def test_opensearch_reranker_request_and_reordering(cfg):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "request_id": "r",
                "latency": 3,
                "usage": {"doc_count": 2},
                "result": {"scores": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}]},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = AliyunOpenSearchReranker(cfg, client=client)
    scores = asyncio.run(reranker.score("等待期多久", ["犹豫期", "等待期为90日"]))
    assert seen["url"] == "https://opensearch.example.com/v3/openapi/workspaces/default/ranker/ops-bge-reranker-larger"
    assert seen["auth"] == "Bearer OS-test"
    assert seen["body"] == {"query": "等待期多久", "docs": ["犹豫期", "等待期为90日"]}
    assert scores == [0.1, 0.9]


def test_opensearch_reranker_truncates_to_512_tokens(cfg):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["docs"] = json.loads(request.content)["docs"]
        return httpx.Response(200, json={"result": {"scores": [{"index": 0, "score": 0.5}]}})

    reranker = AliyunOpenSearchReranker(cfg, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    asyncio.run(reranker.score("问" * 20, ["字" * 900]))
    assert len(seen["docs"][0]) == 512 - 20 - 8


def test_opensearch_reranker_surfaces_api_errors(cfg):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "InvalidParameter", "message": "bad"})

    reranker = AliyunOpenSearchReranker(cfg, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(RuntimeError, match="InvalidParameter"):
        asyncio.run(reranker.score("q", ["d"]))


def test_pai_eas_embedder_batches_and_normalizes(cfg):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((str(request.url), request.headers["Authorization"], body["model"], len(body["input"])))
        data = [{"index": i, "embedding": [3.0, 4.0] + [0.0] * 1022} for i in reversed(range(len(body["input"])))]
        return httpx.Response(200, json={"data": data})

    embedder = AliyunPaiEasEmbedder(cfg, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    vecs = asyncio.run(embedder.embed([f"文本{i}" for i in range(20)]))
    assert vecs.shape == (20, 1024)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert np.allclose(vecs[0][:2], [0.6, 0.8])
    assert [c[3] for c in calls] == [16, 4]
    assert calls[0][:3] == ("https://eas.example.com/api/predict/bge-m3/v1/embeddings", "Bearer eas-test", "bge-m3")


def test_bailian_llm_uses_openai_compatible_endpoint(cfg):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": seen["model"],
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "答案[1]"}}
                ],
            },
        )

    llm = BailianLLM(cfg, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    out = asyncio.run(llm.complete([{"role": "user", "content": "hi"}], model="qwen-plus"))
    assert out == "答案[1]"
    assert seen == {"url": "https://bailian.example.com/compatible-mode/v1/chat/completions", "model": "qwen-plus"}


def test_missing_keys_fail_fast(tmp_path):
    bare = make_settings(tmp_path)
    for cls in (AliyunOpenSearchReranker, AliyunPaiEasEmbedder, BailianLLM):
        with pytest.raises(ValueError, match="未配置"):
            cls(bare)

import asyncio
import logging
from functools import lru_cache
from typing import Protocol

import httpx

from clausetrail.chunking.text_utils import count_tokens, tokenize_for_search
from clausetrail.config import Settings, get_settings

log = logging.getLogger(__name__)

TRUNCATION_MARGIN = 8


class Reranker(Protocol):
    name: str

    async def score(self, query: str, docs: list[str]) -> list[float]: ...


# ops-bge-reranker-larger 限制 query+doc 合计不超过 512 token，超长的 doc 先在本地截断
def truncate_for_pair(query: str, doc: str, max_tokens: int) -> str:
    budget = max_tokens - count_tokens(query) - TRUNCATION_MARGIN
    if budget <= 0 or count_tokens(doc) <= budget:
        return doc
    # 前缀越长 token 数只增不减，二分找放得下的最长前缀，不必每加一个字就把整串重新数一遍
    lo, hi = 0, len(doc)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(doc[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return doc[:lo]


class AliyunOpenSearchReranker:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        if not settings.aliyun_opensearch_host or not settings.aliyun_opensearch_api_key:
            raise ValueError("ALIYUN_OPENSEARCH_HOST / ALIYUN_OPENSEARCH_API_KEY 未配置")
        host = settings.aliyun_opensearch_host.rstrip("/")
        self.name = f"aliyun_opensearch:{settings.aliyun_opensearch_ranker_id}"
        self._url = (
            f"{host}/v3/openapi/workspaces/{settings.aliyun_opensearch_workspace}"
            f"/ranker/{settings.aliyun_opensearch_ranker_id}"
        )
        self._headers = {"Authorization": f"Bearer {settings.aliyun_opensearch_api_key}"}
        self._max_tokens = settings.reranker_max_tokens
        self._client = client or httpx.AsyncClient(timeout=20)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        truncated = [truncate_for_pair(query, d, self._max_tokens) for d in docs]
        cut = sum(1 for a, b in zip(docs, truncated, strict=True) if a != b)
        if cut:
            log.info("rerank: %d/%d 个文档按 %d token 上限截断", cut, len(docs), self._max_tokens)
        resp = await self._client.post(self._url, headers=self._headers, json={"query": query, "docs": truncated})
        resp.raise_for_status()
        body = resp.json()
        if "code" in body and "result" not in body:
            raise RuntimeError(f"OpenSearch ranker 错误: {body.get('code')} {body.get('message')}")
        # 按 index 对回输入位置，不依赖返回顺序；没返回的文档记 0 分
        scores = [0.0] * len(docs)
        for item in body["result"]["scores"]:
            scores[item["index"]] = float(item["score"])
        return scores


class LocalBgeReranker:
    def __init__(self, settings: Settings):
        from FlagEmbedding import FlagReranker

        path = settings.local_model_dir / "bge-reranker-large"
        self.name = f"local:{path}"
        self._model = FlagReranker(str(path), use_fp16=False)
        self._max_tokens = settings.reranker_max_tokens

    async def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        pairs = [[query, d] for d in docs]
        # normalize=True 用 sigmoid 把分数映射到 0–1
        out = await asyncio.to_thread(self._model.compute_score, pairs, normalize=True, max_length=self._max_tokens)
        return [float(s) for s in (out if isinstance(out, list) else [out])]


class FakeReranker:
    name = "fake:lexical"

    async def score(self, query: str, docs: list[str]) -> list[float]:
        from clausetrail.lexicon import query_tokens

        q = {t for t in query_tokens(query).split() if len(t) > 1}
        if not q:
            return [0.0] * len(docs)
        scores = []
        for d in docs:
            terms = set(tokenize_for_search(d))
            scores.append(len(q & terms) / len(q))
        return scores


def build_reranker(settings: Settings) -> Reranker:
    match settings.reranker_backend:
        case "aliyun_opensearch":
            return AliyunOpenSearchReranker(settings)
        case "local":
            return LocalBgeReranker(settings)
        case "fake":
            return FakeReranker()
    raise ValueError(f"未知的 reranker_backend: {settings.reranker_backend}")


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return build_reranker(get_settings())

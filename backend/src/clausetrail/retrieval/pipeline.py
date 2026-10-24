import asyncio
import logging
import time
from dataclasses import dataclass, field

from clausetrail import lexicon
from clausetrail.retrieval.fusion import rrf
from clausetrail.services import Services
from clausetrail.types import ScoredChunk

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    query: str
    candidates: list[ScoredChunk]
    evidence: list[ScoredChunk]
    dense_hits: int
    bm25_hits: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    # 各路召回和融合后的原始排序，评测和调试用
    dense_ids: list[str] = field(default_factory=list)
    bm25_ids: list[str] = field(default_factory=list)
    fused_ids: list[str] = field(default_factory=list)

    @property
    def top_score(self) -> float:
        return max((c.rerank_score or 0.0 for c in self.evidence), default=0.0)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class Retriever:
    def __init__(self, services: Services):
        services.require("embedder", "reranker")
        self.s = services
        self.cfg = services.settings

    async def _dense(self, query: str) -> tuple[list[tuple[str, float]], float, float]:
        t0 = time.perf_counter()
        vec = (await self.s.embedder.embed([query]))[0]
        embed_ms = _ms(t0)
        t1 = time.perf_counter()
        hits = await asyncio.to_thread(self.s.store.search_dense, vec, self.cfg.recall_top_k)
        return hits, embed_ms, _ms(t1)

    async def _bm25(self, query: str) -> tuple[list[tuple[str, float]], float]:
        t0 = time.perf_counter()
        tokens = lexicon.query_tokens(query)
        breaker = self.s.bm25_breaker
        if breaker is None:
            hits = await asyncio.to_thread(self.s.bm25.search, tokens, self.cfg.recall_top_k)
            return hits, _ms(t0)
        hits = None
        if not breaker.is_open():
            try:
                hits = await asyncio.to_thread(self.s.bm25.search, tokens, self.cfg.recall_top_k)
                breaker.success()
            except Exception as exc:
                # 这一问改用 Milvus 内置 BM25。只有连不上、超时、集群不可用才熔断，冷却期内不再每问先等一次超时；
                # 查询或映射写错之类的问题不算故障，熔断了也不会自己好
                log.warning("Elasticsearch 检索失败，这一问改用 Milvus BM25", exc_info=True)
                if self.s.bm25.is_outage(exc):
                    breaker.failure("Elasticsearch 检索失败，暂时改用 Milvus 内置 BM25")
        if hits is None:
            hits = await asyncio.to_thread(self.s.store.search_bm25, tokens, self.cfg.recall_top_k)
        return hits, _ms(t0)

    async def retrieve(self, query: str) -> RetrievalResult:
        t_all = time.perf_counter()
        (dense, embed_ms, dense_ms), (sparse, bm25_ms) = await asyncio.gather(self._dense(query), self._bm25(query))
        dense_ids, bm25_ids = [c for c, _ in dense], [c for c, _ in sparse]
        fused = rrf({"dense": dense_ids, "bm25": bm25_ids}, k=self.cfg.rrf_k)
        # 只用 SQLite 记录当前指向的版本：重新入库时新旧片段会短暂并存（删旧版本失败时会一直留到收尾），
        # 不过滤的话同一条款会出现两次；还没提交的新片段同样不用。
        # 先过滤再取前 rerank_top_k：反过来的话旧片段先占了精排名额，过滤完精排池就不满
        chunks = await asyncio.to_thread(self.s.store.get_chunks, [cid for cid, _, _ in fused])
        current = self.s.db.ready_versions()
        chunks = {cid: c for cid, c in chunks.items() if current.get(c.doc_id) == c.kb_version}
        top = [(cid, score, ranks) for cid, score, ranks in fused if cid in chunks][: self.cfg.rerank_top_k]
        pool = [
            ScoredChunk(
                chunk=chunks[cid],
                score=score,
                dense_rank=ranks.get("dense"),
                bm25_rank=ranks.get("bm25"),
                rrf_score=score,
            )
            for cid, score, ranks in top
        ]
        t_rerank = time.perf_counter()
        # 精排和入库用同一口径（章节标签 + 正文），否则靠章节名召回的片段到精排时分数偏低
        scores = await self.s.reranker.score(query, [c.chunk.search_text for c in pool]) if pool else []
        rerank_ms = _ms(t_rerank)
        for c, s in zip(pool, scores, strict=True):
            c.rerank_score = s
            c.score = s
        pool.sort(key=lambda c: -(c.rerank_score or 0.0))
        return RetrievalResult(
            query=query,
            candidates=pool,
            evidence=pool[: self.cfg.context_top_k],
            dense_hits=len(dense),
            bm25_hits=len(sparse),
            timings_ms={
                "embed": embed_ms,
                "dense": dense_ms,
                "bm25": bm25_ms,
                "rerank": rerank_ms,
                "retrieval": _ms(t_all),
            },
            dense_ids=dense_ids,
            bm25_ids=bm25_ids,
            fused_ids=[cid for cid, _, _ in fused],
        )

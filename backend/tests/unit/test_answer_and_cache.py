import asyncio

from clausetrail.answer.citations import CitationFilter, strip_invalid, used_citations
from clausetrail.answer.prompt import build_messages
from clausetrail.answer.refusal import LOW_SCORE, NO_HIT, precheck
from clausetrail.cache.query_cache import MemoryCache, cache_key, normalize_query
from clausetrail.conversation.continuity import judge
from clausetrail.conversation.rewriter import clean
from clausetrail.models.embedder import FakeEmbedder
from clausetrail.retrieval.fusion import rrf
from clausetrail.retrieval.pipeline import RetrievalResult
from clausetrail.types import Chunk, ScoredChunk


def _scored(score: float, text: str = "等待期为90日。") -> ScoredChunk:
    chunk = Chunk("c1", "d1", "a.pdf", text, ["第五条 等待期"], 3, 3, 0, len(text), "deepdoc")
    return ScoredChunk(chunk=chunk, score=score, rerank_score=score)


def test_rrf_rewards_agreement_between_routes():
    fused = rrf({"dense": ["a", "b", "c"], "bm25": ["b", "d"]}, k=60)
    assert fused[0][0] == "b"
    assert fused[0][2] == {"dense": 2, "bm25": 1}
    assert {cid for cid, _, _ in fused} == {"a", "b", "c", "d"}


def test_citations_are_deduplicated_and_bounded():
    assert used_citations("等待期为90日[1]，免赔额1万元[3][1]。", 3) == [1, 3]
    assert strip_invalid("见[2]和[9]", 3) == "见[2]和"


def test_citation_filter_handles_markers_split_across_deltas():
    f = CitationFilter(2)
    out = "".join(f.feed(d) for d in ["等待期90日[", "1", "]，另见［", "3］。"]) + f.flush()
    assert out == "等待期90日[1]，另见。"
    assert f.text == out


def test_refusal_precheck():
    empty = RetrievalResult("q", [], [], 0, 0)
    assert precheck(empty, 0.3).reason == NO_HIT
    low = RetrievalResult("q", [_scored(0.1)], [_scored(0.1)], 1, 1)
    assert precheck(low, 0.3).reason == LOW_SCORE
    ok = RetrievalResult("q", [_scored(0.9)], [_scored(0.9)], 1, 1)
    assert precheck(ok, 0.3) is None


def test_prompt_numbers_evidence_from_one():
    messages = build_messages("等待期多久", [_scored(0.9)])
    assert "[1] a.pdf · 第五条 等待期 · 第3页" in messages[1]["content"]


def test_cache_key_normalizes_aliases_and_tracks_kb_version():
    assert normalize_query("重疾的观察期多久？") == normalize_query("重大疾病的等待期多久")
    assert cache_key("等待期多久", 1, "m") == cache_key("等待期多久？", 1, "m")
    assert cache_key("等待期多久", 1, "m") != cache_key("等待期多久", 2, "m")
    assert cache_key("等待期多久", 1, "m") != cache_key("等待期多久", 1, "other")


def test_memory_cache_expires():
    async def run():
        live, dead = MemoryCache(ttl=60), MemoryCache(ttl=-1)
        await live.set("k", {"a": 1})
        await dead.set("k", {"a": 1})
        return await live.get("k"), await dead.get("k")

    assert asyncio.run(run()) == ({"a": 1}, None)


def test_continuity_rules():
    emb = FakeEmbedder()

    async def run(q, last, threshold=0.62):
        return await judge(q, last, emb, threshold)

    assert asyncio.run(run("等待期多久", None)).topic == "new"
    d = asyncio.run(run("那它的犹豫期呢", "这款重疾险的等待期多久"))
    assert (d.topic, d.rewrite) == ("continue", True)
    d = asyncio.run(run("意外伤害保险的理赔需要哪些资料", "这款重疾险的等待期多久", threshold=0.95))
    assert (d.topic, d.rewrite) == ("switch", False)


def test_rewrite_output_cleanup():
    assert clean("改写后：「这款重疾险的犹豫期多久」\n多余说明", "原问题") == "这款重疾险的犹豫期多久"
    assert clean("   ", "原问题") == "原问题"

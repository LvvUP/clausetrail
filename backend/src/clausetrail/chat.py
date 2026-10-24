import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from clausetrail.answer import refusal
from clausetrail.answer.citations import CitationFilter, used_citations
from clausetrail.answer.prompt import build_messages
from clausetrail.cache.query_cache import cache_key, config_fingerprint
from clausetrail.conversation import continuity, memory
from clausetrail.conversation.rewriter import rewrite
from clausetrail.retrieval.pipeline import Retriever
from clausetrail.services import Services, ServiceUnavailable
from clausetrail.types import ScoredChunk

log = logging.getLogger(__name__)

Event = tuple[str, dict[str, Any]]
REPLAY_CHUNK_CHARS = 12
TITLE_CHARS = 40


def citation_payload(n: int, sc: ScoredChunk) -> dict[str, Any]:
    c = sc.chunk
    return {
        "n": n,
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "file": c.file_name,
        "section_path": c.section_path,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "snippet": c.text,
        "parser": c.parser,
        "rerank_score": round(sc.rerank_score or 0.0, 4),
        "rrf_score": round(sc.rrf_score, 5),
        "dense_rank": sc.dense_rank,
        "bm25_rank": sc.bm25_rank,
    }


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class ChatService:
    def __init__(self, services: Services):
        self.s = services
        self.cfg = services.settings
        self._retriever: Retriever | None = None
        self._background: set[asyncio.Task] = set()

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.s)
        return self._retriever

    async def stream(self, question: str, session_id: str | None = None) -> AsyncIterator[Event]:
        question = question.strip()
        if not question:
            yield "error", {"code": "empty_question", "message": "问题不能为空"}
            return
        try:
            self.s.require("embedder", "reranker", "llm")
        except ServiceUnavailable as exc:
            yield "error", {"code": "model_unavailable", "message": str(exc)}
            return
        new_session = False
        if session_id:
            session = self.s.db.get_session(session_id)
            if session is None:
                # 会话已被删除或换了库：明确告诉前端，不悄悄新建一个丢掉上下文的会话
                yield "error", {"code": "session_not_found", "message": "这个会话已经不存在了"}
                return
        else:
            session_id = self.s.db.create_session(question[:TITLE_CHARS])
            session = self.s.db.get_session(session_id)
            new_session = True
        turns = self.s.db.turns(session_id)
        record: dict[str, Any] = {
            "question": question,
            "standalone_query": question,
            "topic": continuity.NEW,
            "answer": "",
            "status": "error",
            "refusal_reason": None,
            "citations": [],
            "used_citations": [],
            "cache_hit": False,
            "timings": {},
        }
        started = time.perf_counter()
        turn_id = uuid.uuid4().hex[:16]
        # aclosing 让外层被取消或关闭时内层生成器（连同 LLM 流）一起关闭；
        # 客户端中途断开的这一轮也要落库，状态记为 aborted 后再把异常抛出去
        meta_sent = False
        try:
            run = self._run(question, session_id, turns, session["summary"], record, started, turn_id)
            async with contextlib.aclosing(run) as events:
                async for event in events:
                    meta_sent = meta_sent or event[0] == "meta"
                    yield event
        except (asyncio.CancelledError, GeneratorExit):
            # 结果已经定下（拒答，或回答已经完整生成、只差写缓存）之后才断开的，保留原状态，不改成 aborted
            if record["status"] == "error":
                record["status"] = "aborted"
            self._finish_early(session_id, turn_id, record, started, meta_sent, new_session)
            raise
        except Exception:
            log.exception("chat failed: session=%s", session_id)
            record["status"] = "error"
            self._finish_early(session_id, turn_id, record, started, meta_sent, new_session)
            yield "error", {"code": "internal", "message": "处理问题时出错了，请稍后再试"}
            return
        self._save(session_id, turn_id, record, started)
        yield (
            "done",
            {
                "session_id": session_id,
                "message_id": turn_id,
                "status": record["status"],
                "refusal_reason": record["refusal_reason"],
                "used_citations": record["used_citations"],
                "cache_hit": record["cache_hit"],
                "timings_ms": record["timings"],
            },
        )
        self._schedule_summary(session_id)

    async def _run(self, question, session_id, turns, summary, record, started, turn_id) -> AsyncIterator[Event]:
        cfg, timings = self.cfg, record["timings"]
        # 和上一轮改写后的独立问句比较，上一轮的原话本身也可能是省略句；
        # 出错 / 中止的轮次不算，和改写窗口（memory.window）同一口径
        usable = memory.window(turns, 1)
        last = usable[-1]["standalone_query"] if usable else None
        t0 = time.perf_counter()
        standalone = question
        try:
            decision = await continuity.judge(question, last, self.s.embedder, cfg.continuity_similarity_threshold)
            if decision.rewrite:
                window = memory.window(turns, cfg.history_window_turns)
                standalone = await rewrite(self.s.llm, cfg.llm_rewrite_model, question, window, summary)
        except Exception:
            # 话题判断或改写失败时不中断这一问，按新话题、用原问题检索
            log.warning("话题判断或改写失败，按原问题检索: session=%s", session_id, exc_info=True)
            decision = continuity.Decision(continuity.NEW, False, "rewrite_failed")
            standalone = question
        timings["rewrite"] = _ms(t0)
        record["standalone_query"], record["topic"] = standalone, decision.topic

        kb_version = self.s.db.kb_version()
        key = cache_key(standalone, kb_version, config_fingerprint(cfg, self.s.db.kb_epoch()))
        cached = await self._cache_get(key)
        yield (
            "meta",
            {
                "session_id": session_id,
                "message_id": turn_id,
                "question": question,
                "rewritten_query": standalone if standalone != question else None,
                "topic": decision.topic,
                "topic_reason": decision.reason,
                "similarity": decision.similarity,
                "cache_hit": cached is not None,
                "kb_version": kb_version,
            },
        )

        # 命中缓存时按实时生成的事件顺序（retrieval 再分段 delta）回放，前端不用区分来源
        if cached:
            record.update(
                answer=cached["answer"],
                citations=cached["citations"],
                used_citations=cached["used_citations"],
                cache_hit=True,
            )
            yield (
                "retrieval",
                {
                    "citations": cached["citations"],
                    "cached": True,
                    "candidates": len(cached["citations"]),
                    "dense_hits": None,
                    "bm25_hits": None,
                    "bm25_backend": self.s.bm25.name,
                    "top_score": max((c["rerank_score"] for c in cached["citations"]), default=0.0),
                    "timings_ms": {},
                },
            )
            for i in range(0, len(cached["answer"]), REPLAY_CHUNK_CHARS):
                timings.setdefault("first_token", _ms(started))
                yield "delta", {"text": cached["answer"][i : i + REPLAY_CHUNK_CHARS]}
            # 回放完才算答完：回放到一半断开的记为 aborted
            record["status"] = "answered"
            return

        result = await self.retriever.retrieve(standalone)
        timings.update(result.timings_ms)
        citations = [citation_payload(i, sc) for i, sc in enumerate(result.evidence, start=1)]
        record["citations"] = citations
        yield (
            "retrieval",
            {
                "citations": citations,
                "cached": False,
                "candidates": len(result.candidates),
                "dense_hits": result.dense_hits,
                "bm25_hits": result.bm25_hits,
                "bm25_backend": self.s.bm25.name,
                "top_score": round(result.top_score, 4),
                "timings_ms": result.timings_ms,
            },
        )

        pre = refusal.precheck(result, cfg.refusal_threshold)
        if pre:
            record.update(status="refused", refusal_reason=pre.reason)
            yield "refusal", {"reason": pre.reason, "message": pre.message, "top_score": round(pre.top_score, 4)}
            return

        t1 = time.perf_counter()
        filt = CitationFilter(len(result.evidence))
        llm_stream = self.s.llm.stream(build_messages(standalone, result.evidence), model=cfg.llm_answer_model)
        # 客户端断开、这个生成器被关闭时，显式关掉 LLM 流和底层 HTTP 响应，不等垃圾回收
        async with contextlib.aclosing(llm_stream) as deltas:
            async for delta in deltas:
                out = filt.feed(delta)
                if out:
                    timings.setdefault("first_token", _ms(started))
                    # 边生成边记下来：用户中途停止时，已经输出的部分也会落库
                    record["answer"] = filt.text
                    record["used_citations"] = used_citations(filt.text, len(result.evidence))
                    yield "delta", {"text": out}
        tail = filt.flush()
        if tail:
            yield "delta", {"text": tail}
        timings["generate"] = _ms(t1)
        record["answer"] = filt.text
        record["used_citations"] = used_citations(filt.text, len(result.evidence))
        if not record["used_citations"]:
            post = refusal.unsupported(result.top_score)
            # 已经流出去的文字没有出处：前端收到 refusal 就撤下这段文字，库里也不存，刷新后看到的和当时一致
            record.update(status="refused", refusal_reason=post.reason, answer="")
            yield "refusal", {"reason": post.reason, "message": post.message, "top_score": round(post.top_score, 4)}
            return
        record["status"] = "answered"
        # 只缓存通过引用校验的回答；拒答在前面已经提前返回，不会进缓存
        await self._cache_set(
            key, {"answer": record["answer"], "citations": citations, "used_citations": record["used_citations"]}
        )

    async def _cache_get(self, key: str) -> dict | None:
        try:
            return await self.s.cache.get(key)
        except Exception:
            log.warning("读取缓存失败", exc_info=True)
            return None

    async def _cache_set(self, key: str, value: dict) -> None:
        try:
            await self.s.cache.set(key, value)
        except Exception:
            log.warning("写入缓存失败", exc_info=True)

    # meta 之前前端还拿不到 message_id，这时落库会和前端的临时回合对不上，出现两份；
    # 首页新建的会话还会在侧栏多出一条空会话。所以 meta 之前结束的这一轮不落库，新建的空会话直接删掉
    def _finish_early(
        self, session_id: str, turn_id: str, record: dict, started: float, meta_sent: bool, new_session: bool
    ) -> None:
        if meta_sent:
            self._save(session_id, turn_id, record, started)
        elif new_session:
            try:
                self.s.db.delete_session(session_id)
            except Exception:
                log.warning("删除空会话失败: session=%s", session_id, exc_info=True)

    def _save(self, session_id: str, turn_id: str, record: dict, started: float) -> None:
        record["timings"]["total"] = _ms(started)
        try:
            self.s.db.add_turn(session_id, record, turn_id=turn_id)
        except Exception:
            log.exception("保存对话失败: session=%s", session_id)

    # 事件循环只持有 task 的弱引用，放进集合里免得摘要任务跑到一半被回收；drain 时统一等待
    def _schedule_summary(self, session_id: str) -> None:
        task = asyncio.create_task(self._summarize(session_id))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _summarize(self, session_id: str) -> None:
        try:
            await memory.maybe_summarize(
                self.s.db,
                self.s.llm,
                self.cfg.llm_rewrite_model,
                session_id,
                self.cfg.history_window_turns,
                self.cfg.summary_max_chars,
            )
        except Exception:
            log.warning("会话摘要失败: session=%s", session_id, exc_info=True)

    async def drain(self) -> None:
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)

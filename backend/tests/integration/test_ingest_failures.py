import asyncio
import json
import os
import time

import pytest

from clausetrail.breaker import Breaker
from clausetrail.config import PROJECT_ROOT
from clausetrail.index import pending
from clausetrail.ingest.pipeline import ingest_file, remove_document
from clausetrail.retrieval.pipeline import Retriever

WHOLE_LIFE = PROJECT_ROOT / "data" / "samples" / "chinalife-whole-life-hengyu-yisheng.pdf"


def _chunk_ids(services, doc_id: str) -> set[str]:
    return {cid for cid, _, _ in services.store.document_tokens(doc_id)}


def _boom(*args, **kwargs):
    raise RuntimeError("模拟写入失败")


def test_failed_reingest_keeps_previous_version(services, ingested, monkeypatch):
    before = _chunk_ids(services, ingested.doc_id)
    monkeypatch.setattr(services.bm25, "index", _boom)
    with pytest.raises(RuntimeError):
        asyncio.run(ingest_file(services, WHOLE_LIFE))
    assert _chunk_ids(services, ingested.doc_id) == before
    assert services.db.get_document(ingested.doc_id)["status"] == "ready"
    assert pending.pending(services.db) == {}


def test_reingest_replaces_chunks_without_duplicates(services, ingested):
    before = _chunk_ids(services, ingested.doc_id)
    report = asyncio.run(ingest_file(services, WHOLE_LIFE))
    after = _chunk_ids(services, ingested.doc_id)
    assert len(after) == report.chunks == len(before)
    assert after.isdisjoint(before)
    assert services.db.get_document(ingested.doc_id)["kb_version"] == report.kb_version


def test_recover_keeps_committed_version_after_interrupted_write(services, ingested):
    before = _chunk_ids(services, ingested.doc_id)
    # 模拟写新版本时进程被杀：pending 里记着新版本，SQLite 还指向旧版本
    pending.mark(services.db, services.store, ingested.doc_id, services.db.kb_version() + 1)
    assert pending.recover(services.db, services.store) == 1
    assert _chunk_ids(services, ingested.doc_id) == before
    assert pending.pending(services.db) == {}


def test_leftover_old_version_is_not_retrieved_and_gets_cleaned(services, ingested, monkeypatch):
    before = _chunk_ids(services, ingested.doc_id)
    real_delete = services.store.delete_document

    # 只让「提交后删旧版本」这一步失败
    def fail_old_delete(doc_id, keep_version=None, only_version=None):
        if keep_version is not None:
            raise RuntimeError("模拟删除旧版本失败")
        return real_delete(doc_id, keep_version, only_version)

    monkeypatch.setattr(services.store, "delete_document", fail_old_delete)
    asyncio.run(ingest_file(services, WHOLE_LIFE))
    monkeypatch.undo()
    version = services.db.get_document(ingested.doc_id)["kb_version"]
    assert before < _chunk_ids(services, ingested.doc_id)
    result = asyncio.run(Retriever(services).retrieve("犹豫期是多久"))
    assert all(c.chunk.kb_version == version for c in result.candidates)
    # 旧片段先被过滤，不占精排名额：精排池照样是满的
    assert len(result.candidates) == services.settings.rerank_top_k
    # 记录已经指向新版本：recover 走「只删旧版本」这一支
    assert pending.pending(services.db) == {ingested.doc_id: version}
    assert pending.recover(services.db, services.store) == 1
    assert _chunk_ids(services, ingested.doc_id).isdisjoint(before)
    assert pending.pending(services.db) == {}


def test_failed_cleanup_is_finished_by_recover(services, ingested, monkeypatch):
    before = _chunk_ids(services, ingested.doc_id)
    real_delete = services.store.delete_document

    # 写入失败，清理新片段这一步也失败：pending 留着，等 recover 收尾
    def fail_new_delete(doc_id, keep_version=None, only_version=None):
        if only_version is not None:
            raise RuntimeError("模拟清理失败")
        return real_delete(doc_id, keep_version, only_version)

    monkeypatch.setattr(services.bm25, "index", _boom)
    monkeypatch.setattr(services.store, "delete_document", fail_new_delete)
    with pytest.raises(RuntimeError):
        asyncio.run(ingest_file(services, WHOLE_LIFE))
    monkeypatch.undo()
    assert before < _chunk_ids(services, ingested.doc_id)
    assert services.db.get_document(ingested.doc_id)["status"] == "ready"
    assert pending.recover(services.db, services.store) == 1
    assert _chunk_ids(services, ingested.doc_id) == before
    assert pending.pending(services.db) == {}


def test_write_in_progress_in_another_process_is_left_alone(services, ingested):
    version = services.db.kb_version() + 1
    # 父进程一定还活着，拿它的 pid 模拟另一个正在入库的进程
    entry = {"version": version, "pid": os.getppid(), "at": time.time()}
    services.db.set_meta(pending.KEY_PREFIX + ingested.doc_id, json.dumps(entry))
    try:
        assert pending.recover(services.db, services.store) == 0
        assert pending.pending(services.db) == {ingested.doc_id: version}
        with pytest.raises(pending.IngestInProgress):
            asyncio.run(ingest_file(services, WHOLE_LIFE))
    finally:
        pending.clear(services.db, ingested.doc_id)


class _FlakyES:
    name = "elasticsearch"

    def __init__(self, outage: bool):
        self.outage = outage

    def search(self, query_tokens: str, k: int):
        raise RuntimeError("模拟 ES 出错")

    def is_outage(self, exc: BaseException) -> bool:
        return self.outage


@pytest.mark.parametrize("outage", [False, True])
def test_es_errors_fall_back_and_only_outages_trip_breaker(services, ingested, monkeypatch, outage):
    breaker = Breaker("bm25", {})
    monkeypatch.setattr(services, "bm25", _FlakyES(outage))
    monkeypatch.setattr(services, "bm25_breaker", breaker)
    result = asyncio.run(Retriever(services).retrieve("犹豫期是多久"))
    assert result.bm25_hits > 0
    assert breaker.is_open() is outage


def test_remove_can_be_retried_after_failure(services, ingested, monkeypatch):
    monkeypatch.setattr(services.bm25, "delete_document", _boom)
    with pytest.raises(RuntimeError):
        asyncio.run(remove_document(services, ingested.doc_id))
    assert services.db.get_document(ingested.doc_id) is not None
    monkeypatch.undo()
    version = services.db.kb_version()
    assert asyncio.run(remove_document(services, ingested.doc_id)) is True
    # 删除前后各加一次版本号
    assert services.db.kb_version() == version + 2
    assert services.db.get_document(ingested.doc_id) is None
    assert _chunk_ids(services, ingested.doc_id) == set()

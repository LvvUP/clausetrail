import logging
from typing import Protocol

from clausetrail import lexicon
from clausetrail.config import Settings, redact_url
from clausetrail.index.milvus_store import MilvusStore
from clausetrail.storage.db import Database
from clausetrail.types import Chunk

log = logging.getLogger(__name__)

ES_SYNC_KEY = "es_synced_kb_version"
ES_LEXICON_KEY = "es_lexicon_fingerprint"


class BM25Backend(Protocol):
    name: str

    def index(self, chunks: list[Chunk], tokens: list[str]) -> None: ...

    def delete_document(self, doc_id: str, keep_ids: list[str] | None = None) -> None: ...

    def delete_chunks(self, chunk_ids: list[str]) -> None: ...

    def search(self, query_tokens: str, k: int) -> list[tuple[str, float]]: ...

    # 检索抛出的异常是不是后端故障（连不上、超时），决定要不要熔断
    def is_outage(self, exc: BaseException) -> bool: ...


# BM25 和向量共用同一个 Milvus 集合，写入和删除已由 MilvusStore.insert_chunks / delete_document 完成，
# 所以这里的写入和删除什么都不做
class MilvusBM25:
    name = "milvus_bm25"

    def __init__(self, store: MilvusStore):
        self._store = store

    def index(self, chunks: list[Chunk], tokens: list[str]) -> None:
        return None

    def delete_document(self, doc_id: str, keep_ids: list[str] | None = None) -> None:
        return None

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        return None

    def search(self, query_tokens: str, k: int) -> list[tuple[str, float]]:
        return self._store.search_bm25(query_tokens, k)

    def is_outage(self, exc: BaseException) -> bool:
        return False


def build_bm25(settings: Settings, store: MilvusStore) -> tuple[BM25Backend, str | None]:
    if settings.bm25_backend == "milvus":
        return MilvusBM25(store), None
    try:
        from clausetrail.index.es_store import ElasticsearchBM25
    except ImportError:
        reason = "elasticsearch 客户端未安装（pip install '.[es]'），BM25 降级为 Milvus 内置 BM25"
        log.warning(reason)
        return MilvusBM25(store), reason
    es = ElasticsearchBM25(settings)
    if es.available():
        return es, None
    reason = f"连不上 Elasticsearch（{redact_url(settings.es_url)}），BM25 降级为 Milvus 内置 BM25"
    log.warning(reason)
    return MilvusBM25(store), reason


def mark_synced(bm25: BM25Backend, db: Database) -> None:
    """ES 写入成功后记下它对应的知识库版本；降级为 Milvus BM25 时不记，下次 ES 可用时会重建。"""
    if bm25.name == "elasticsearch":
        db.set_meta(ES_SYNC_KEY, str(db.kb_version()))


def sync_es(es, db: Database, store: MilvusStore) -> None:
    """启动时检查 ES 是否落后于知识库（降级期间入库或删除过文档、索引被清空、词表改过），落后就重建。"""
    # 同义词在建索引时写死在分析器里，词表改了只能整个重建，所以词表指纹也要对得上
    lexicon_ok = db.get_meta(ES_LEXICON_KEY) == lexicon.fingerprint()
    if lexicon_ok and db.get_meta(ES_SYNC_KEY) == str(db.kb_version()) and es.has_index():
        return
    rows = [row for d in db.list_documents() for row in store.document_tokens(d["doc_id"])]
    es.rebuild(rows)
    db.set_meta(ES_SYNC_KEY, str(db.kb_version()))
    db.set_meta(ES_LEXICON_KEY, lexicon.fingerprint())
    log.info("elasticsearch 已按 Milvus 重建：%d 个片段", len(rows))

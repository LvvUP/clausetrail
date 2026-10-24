import logging

from clausetrail import lexicon
from clausetrail.config import Settings
from clausetrail.types import Chunk

log = logging.getLogger(__name__)


class ElasticsearchBM25:
    name = "elasticsearch"

    def __init__(self, settings: Settings):
        from elasticsearch import Elasticsearch

        auth = (settings.es_username, settings.es_password) if settings.es_username else None
        self._es = Elasticsearch(settings.es_url, basic_auth=auth, request_timeout=10)
        self._index = settings.es_index

    def available(self) -> bool:
        try:
            return bool(self._es.ping())
        except Exception:
            return False

    # 同义词两边都写成索引里的分词形式（空格分隔），否则 synonym_graph 对不上 whitespace 切出来的词
    def _synonym_rules(self) -> list[str]:
        rules = set()
        for alias, canonical in lexicon.aliases().items():
            a, c = " ".join(lexicon.index_form(alias)), " ".join(lexicon.index_form(canonical))
            if a != c:
                rules.add(f"{a}, {c}")
        return sorted(rules)

    # 同义词只在建索引时写进分析器，索引已存在就直接返回；改了词表后由启动时的 sync_es 按词表指纹重建
    def ensure_index(self) -> None:
        if self._es.indices.exists(index=self._index):
            return
        self._es.indices.create(
            index=self._index,
            settings={
                "analysis": {
                    "filter": {"clause_synonyms": {"type": "synonym_graph", "synonyms": self._synonym_rules()}},
                    "analyzer": {
                        "pretokenized": {"type": "custom", "tokenizer": "whitespace", "filter": ["lowercase"]},
                        "pretokenized_synonyms": {
                            "type": "custom",
                            "tokenizer": "whitespace",
                            "filter": ["lowercase", "clause_synonyms"],
                        },
                    },
                },
            },
            mappings={
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    # tokens 已经是 jieba 分好、按空格分隔的词，ES 只按空格切；synonym_graph 只能用在查询端
                    "tokens": {"type": "text", "analyzer": "pretokenized", "search_analyzer": "pretokenized_synonyms"},
                }
            },
        )
        log.info("created elasticsearch index %s with %d synonym rules", self._index, len(self._synonym_rules()))

    def index(self, chunks: list[Chunk], tokens: list[str]) -> None:
        from elasticsearch import helpers

        self.ensure_index()
        actions = [
            {
                "_index": self._index,
                "_id": c.chunk_id,
                "_source": {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "tokens": t},
            }
            for c, t in zip(chunks, tokens, strict=True)
        ]
        helpers.bulk(self._es, actions, refresh=True)

    # 索引还不存在时 delete_by_query 会报 index_not_found，直接跳过。
    # keep_ids 是重新入库时刚写入的新片段：删旧版本时保留它们
    def delete_document(self, doc_id: str, keep_ids: list[str] | None = None) -> None:
        if not self._es.indices.exists(index=self._index):
            return
        query: dict = {"term": {"doc_id": doc_id}}
        if keep_ids:
            query = {"bool": {"filter": [query], "must_not": [{"ids": {"values": keep_ids}}]}}
        self._es.delete_by_query(index=self._index, query=query, refresh=True)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if chunk_ids and self._es.indices.exists(index=self._index):
            self._es.delete_by_query(index=self._index, query={"ids": {"values": chunk_ids}}, refresh=True)

    def has_index(self) -> bool:
        return bool(self._es.indices.exists(index=self._index))

    # 降级期间入库或删除的文档只写进了 Milvus；ES 落后于知识库时，用 Milvus 里存的分词结果整体重建
    def rebuild(self, rows: list[tuple[str, str, str]]) -> None:
        from elasticsearch import helpers

        if self.has_index():
            self._es.indices.delete(index=self._index)
        self.ensure_index()
        actions = [
            {"_index": self._index, "_id": cid, "_source": {"chunk_id": cid, "doc_id": did, "tokens": tokens}}
            for cid, did, tokens in rows
        ]
        if actions:
            helpers.bulk(self._es, actions, refresh=True)

    # 连不上、超时（TransportError 及其子类）和集群返回 5xx / 429 才算 ES 故障；4xx 多半是查询或映射的问题
    @staticmethod
    def is_outage(exc: BaseException) -> bool:
        from elasticsearch import ApiError, TransportError

        if isinstance(exc, TransportError):
            return True
        if isinstance(exc, ApiError):
            status = getattr(exc, "status_code", 0) or 0
            return status >= 500 or status == 429
        return False

    def search(self, query_tokens: str, k: int) -> list[tuple[str, float]]:
        from elasticsearch import NotFoundError

        if not query_tokens.strip():
            return []
        try:
            res = self._es.search(index=self._index, query={"match": {"tokens": query_tokens}}, size=k, _source=False)
        except NotFoundError:
            # 还没入过库，索引不存在，等同于没有命中
            return []
        return [(h["_id"], float(h["_score"])) for h in res["hits"]["hits"]]

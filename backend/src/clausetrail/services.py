import logging
from dataclasses import dataclass, field

from clausetrail import lexicon
from clausetrail.breaker import Breaker
from clausetrail.cache.query_cache import QueryCache, build_cache
from clausetrail.config import Settings, get_settings
from clausetrail.index.bm25 import BM25Backend, MilvusBM25, build_bm25, sync_es
from clausetrail.index.milvus_store import MilvusStore
from clausetrail.index.pending import recover as recover_pending
from clausetrail.models.embedder import Embedder, build_embedder
from clausetrail.models.llm import LLM, build_llm
from clausetrail.models.reranker import Reranker, build_reranker
from clausetrail.storage.db import Database

log = logging.getLogger(__name__)


class ServiceUnavailable(RuntimeError):
    def __init__(self, component: str, reason: str):
        super().__init__(f"{component} 不可用：{reason}")
        self.component = component
        self.reason = reason


@dataclass
class Services:
    settings: Settings
    db: Database
    store: MilvusStore
    bm25: BM25Backend
    cache: QueryCache
    embedder: Embedder | None
    reranker: Reranker | None
    llm: LLM | None
    # errors 记组件不可用的原因；degraded 记组件改用降级实现（BM25、缓存）运行的原因
    errors: dict[str, str] = field(default_factory=dict)
    degraded: dict[str, str] = field(default_factory=dict)
    # 只有 BM25 用 ES 时才有：ES 运行中出故障时由它切到 Milvus BM25
    bm25_breaker: Breaker | None = None

    def require(self, *components: str) -> None:
        for name in components:
            if getattr(self, name) is None:
                raise ServiceUnavailable(name, self.errors.get(name, "未初始化"))

    # 模型客户端和 Redis 都是异步客户端，要在事件循环结束前关闭
    async def aclose(self) -> None:
        for component in (self.embedder, self.reranker, self.llm, self.cache):
            close = getattr(component, "aclose", None)
            if close is None:
                continue
            try:
                await close()
            except Exception:
                log.warning("关闭 %s 失败", type(component).__name__, exc_info=True)

    def close(self) -> None:
        self.store.close()
        self.db.close()


# 模型客户端构建失败（多半是没配密钥）不阻止启动，先记下原因，用到时由 require 报出
def _try(builder, settings: Settings, name: str, errors: dict[str, str]):
    try:
        return builder(settings)
    except Exception as exc:
        errors[name] = str(exc)
        log.warning("%s 未就绪：%s", name, exc)
        return None


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    lexicon.ensure_loaded()
    errors: dict[str, str] = {}
    degraded: dict[str, str] = {}
    embedder = _try(build_embedder, settings, "embedder", errors)
    reranker = _try(build_reranker, settings, "reranker", errors)
    llm = _try(build_llm, settings, "llm", errors)
    # embedder 没就绪时按配置的维度打开集合，服务照样能启动
    dim = embedder.dim if embedder else settings.embedding_dim
    store = MilvusStore(
        settings.vector_store_uri,
        settings.vector_collection,
        dim,
        settings.hnsw_m,
        settings.hnsw_ef_construction,
        settings.hnsw_ef,
    )
    db = None
    # 建到一半失败（如 SQLite 打不开）时，先关掉已经打开的连接再抛出
    try:
        db = Database(settings.sqlite_path)
        # 上次入库中途被杀时先收尾，再按收尾后的状态同步 ES
        recover_pending(db, store)
        bm25, bm25_reason = build_bm25(settings, store)
        if bm25_reason is None and bm25.name == "elasticsearch":
            try:
                sync_es(bm25, db, store)
            except Exception as exc:
                bm25_reason = f"Elasticsearch 同步失败（{type(exc).__name__}），BM25 降级为 Milvus 内置 BM25"
                log.warning(bm25_reason, exc_info=True)
                bm25 = MilvusBM25(store)
        if bm25_reason:
            degraded["bm25"] = bm25_reason
        cache, cache_reason = build_cache(settings, degraded)
        if cache_reason:
            degraded["cache"] = cache_reason
    except Exception:
        if db is not None:
            db.close()
        store.close()
        raise
    return Services(
        settings=settings,
        db=db,
        store=store,
        bm25=bm25,
        cache=cache,
        embedder=embedder,
        reranker=reranker,
        llm=llm,
        errors=errors,
        degraded=degraded,
        bm25_breaker=Breaker("bm25", degraded) if bm25.name == "elasticsearch" else None,
    )

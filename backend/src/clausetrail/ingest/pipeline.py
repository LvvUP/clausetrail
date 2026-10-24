import asyncio
import hashlib
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from clausetrail import lexicon
from clausetrail.chunking.splitter import chunk_document
from clausetrail.index import pending
from clausetrail.index.bm25 import mark_synced
from clausetrail.index.milvus_store import _check_doc_id
from clausetrail.ingest.router import IMAGE_SUFFIXES, body_blocks, parse_document
from clausetrail.services import Services
from clausetrail.storage.db import now_iso
from clausetrail.structure.heading_classifier import assign_levels
from clausetrail.structure.ocr_correction import correct_blocks
from clausetrail.structure.tree import build_tree
from clausetrail.types import Chunk

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", *IMAGE_SUFFIXES}
# 切分线程等主循环里的 embed 结果，最长等这么久，避免主循环卡住时线程永远挂起
EMBED_TIMEOUT_SECONDS = 300


@dataclass
class IngestReport:
    doc_id: str
    file_name: str
    pages: int
    chunks: int
    page_types: dict[str, int]
    parsers: dict[str, int]
    ocr_fixes: int
    kb_version: int
    seconds: float

    def as_dict(self) -> dict:
        return asdict(self)


# doc_id 取文件内容的哈希：同一份文件重复入库（哪怕改了名）会覆盖旧数据，不会多出一份
def doc_id_for(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def embedding_text(chunk: Chunk) -> str:
    return chunk.search_text


# 写到一半失败时只清掉这次写入的新片段：重新入库失败不影响已经可用的旧版本，只有首次入库才把文档记为 failed。
# 版本号照样加一，写入期间可能有问答引用了这些新片段
async def _discard(services: Services, doc_id: str, version: int, chunk_ids: list[str], record: dict) -> None:
    cleaned = True
    try:
        await asyncio.to_thread(services.store.delete_document, doc_id, None, version)
        await asyncio.to_thread(services.bm25.delete_chunks, chunk_ids)
    except Exception:
        cleaned = False
        log.warning("清理失败入库的残留数据出错，下次启动时再清：doc=%s", doc_id, exc_info=True)
    existing = services.db.get_document(doc_id)
    if existing is None or existing.get("status") != "ready":
        services.db.upsert_document({**record, "chunks": 0, "status": "failed", "kb_version": services.db.kb_version()})
    if cleaned:
        pending.clear(services.db, doc_id)
    services.db.bump_kb_version()


async def ingest_file(services: Services, path: Path) -> IngestReport:
    services.require("embedder")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型：{path.name}")
    settings = services.settings
    started = time.perf_counter()
    loop = asyncio.get_running_loop()
    doc_id = doc_id_for(path)

    # 切分在工作线程里跑，而 embedder 的异步客户端属于主事件循环，所以把协程投回主循环再同步等结果
    def embed_from_worker(texts: list[str]):
        return asyncio.run_coroutine_threadsafe(services.embedder.embed(texts), loop).result(EMBED_TIMEOUT_SECONDS)

    def parse_and_chunk():
        doc = parse_document(path, use_layout=settings.layout_analysis)
        blocks = body_blocks(doc)
        fixes = correct_blocks(blocks)
        assign_levels(blocks)
        chunks = chunk_document(
            build_tree(blocks),
            doc_id,
            path.name,
            parser="deepdoc",
            max_tokens=settings.chunk_max_tokens,
            overlap_sentences=settings.chunk_overlap_sentences,
            embed=embed_from_worker,
            similarity_threshold=settings.merge_similarity_threshold,
        )
        return doc, chunks, fixes

    doc, chunks, fixes = await asyncio.to_thread(parse_and_chunk)
    if not chunks:
        raise ValueError(f"{path.name} 没有解析出可检索的内容")

    next_version = services.db.kb_version() + 1
    for c in chunks:
        c.kb_version = next_version
        # chunk_id 带上版本：新旧片段在两路索引里可以并存，旧版本等新版本提交后再删
        c.chunk_id = hashlib.sha1(f"{c.chunk_id}|{next_version}".encode()).hexdigest()[:16]
    texts = [embedding_text(c) for c in chunks]
    vectors = await services.embedder.embed(texts)
    tokens = [lexicon.index_tokens(t) for t in texts]
    page_types = dict(Counter(t.value for t in doc.page_types.values()))
    parsers = dict(doc.parsers)
    record = {
        "doc_id": doc_id,
        "file_name": path.name,
        "path": str(path.resolve()),
        "pages": doc.page_count,
        "chunks": len(chunks),
        "page_types": page_types,
        "parsers": parsers,
        "status": "ready",
        "kb_version": next_version,
        "ingested_at": now_iso(),
    }
    new_ids = [c.chunk_id for c in chunks]
    # 另一个进程正在入库同一份文档时直接报错；上一次没收尾的残留在这里先清掉
    await asyncio.to_thread(pending.mark, services.db, services.store, doc_id, next_version)
    try:
        await asyncio.to_thread(services.store.insert_chunks, chunks, vectors, tokens)
        await asyncio.to_thread(services.bm25.index, chunks, tokens)
    except Exception:
        await _discard(services, doc_id, next_version, new_ids, record)
        raise
    # 新片段两路都写好才提交。先让记录指向新版本（检索只认记录指向的版本，见 retrieval.pipeline），再把版本号加一：
    # 写入期间的问答都记在旧版本的缓存 key 下，加一后自然失效；反过来先加一的话，中间那一下会把旧片段缓存到新 key 下
    services.db.set_meta("embedder", services.embedder.name)
    services.db.upsert_document(record)
    kb_version = services.db.bump_kb_version()
    # 提交之后再删旧版本；这一步失败时旧片段不会被检索到，pending 记录留着，
    # 由下次启动的 pending.recover 或下次入库这份文档时收尾
    try:
        await asyncio.to_thread(services.store.delete_document, doc_id, next_version)
        await asyncio.to_thread(services.bm25.delete_document, doc_id, new_ids)
    except Exception:
        log.warning("删除旧版本片段失败，之后会自动清理：doc=%s", doc_id, exc_info=True)
    else:
        pending.clear(services.db, doc_id)
        mark_synced(services.bm25, services.db)
    report = IngestReport(
        doc_id,
        path.name,
        doc.page_count,
        len(chunks),
        page_types,
        parsers,
        len(fixes),
        kb_version,
        round(time.perf_counter() - started, 1),
    )
    log.info("ingested %s", report.as_dict())
    return report


# 先加版本号，让引用这份文档的缓存立刻失效；再删两路索引，最后删 SQLite 记录。
# 中途失败时记录还在、可以重跑 remove（每一步删除都能重复执行）；即使 SQLite 里没有记录，索引残留也会一并删掉。
# 删完再加一次：删除期间的问答可能还检索到这份文档，结果记在第一次加出来的版本下，不能留着
async def remove_document(services: Services, doc_id: str) -> bool:
    _check_doc_id(doc_id)
    existed = services.db.get_document(doc_id) is not None
    services.db.bump_kb_version()
    await asyncio.to_thread(services.store.delete_document, doc_id)
    await asyncio.to_thread(services.bm25.delete_document, doc_id)
    services.db.delete_document(doc_id)
    pending.clear(services.db, doc_id)
    services.db.bump_kb_version()
    mark_synced(services.bm25, services.db)
    return existed

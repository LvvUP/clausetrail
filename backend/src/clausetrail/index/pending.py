import json
import logging
import os
import sys
import time

from clausetrail.index.milvus_store import MilvusStore
from clausetrail.storage.db import Database

log = logging.getLogger(__name__)

KEY_PREFIX = "pending_write:"
# 超过这么久还没收尾的记录按中断处理，不再认为写入它的进程还在写（防 pid 被系统复用后永远跳过）
STALE_SECONDS = 6 * 3600


class IngestInProgress(RuntimeError):
    pass


# 入库分几步写（新片段 → SQLite 记录 → 删旧片段）。开始前按 doc_id 记下「哪个进程在写哪个版本」，全部完成后清掉；
# 进程中途被杀时，由下次启动的 recover 或下一次入库同一份文档时收尾，不留孤儿片段，也不丢旧版本。
# 每份文档一个 meta 键，标记和清除都是单键写入，不会像整表读改写那样被别的进程覆盖
def _key(doc_id: str) -> str:
    return KEY_PREFIX + doc_id


def _entries(db: Database) -> dict[str, dict]:
    return {k[len(KEY_PREFIX) :]: json.loads(v) for k, v in db.meta_items(KEY_PREFIX).items()}


def pending(db: Database) -> dict[str, int]:
    return {doc_id: int(e["version"]) for doc_id, e in _entries(db).items()}


def _alive(pid: int) -> bool:
    # Windows 上 os.kill 会直接结束进程，不能拿来探测；这时只按 STALE_SECONDS 判断
    if sys.platform == "win32":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# 别的进程正在写：独立部署的 Milvus 允许 API 和 CLI 同时运行，API 启动时不能把 CLI 刚写进去的片段当成残留删掉
def _in_flight(entry: dict) -> bool:
    pid = int(entry.get("pid", 0))
    if pid in (0, os.getpid()):
        return False
    if time.time() - float(entry.get("at", 0)) > STALE_SECONDS:
        return False
    return _alive(pid)


def _settle(db: Database, store: MilvusStore, doc_id: str, version: int) -> None:
    record = db.get_document(doc_id)
    if record and record.get("status") == "ready" and int(record["kb_version"]) == version:
        # SQLite 已经指向这个版本：它是有效数据，只差删掉更早的版本
        store.delete_document(doc_id, keep_version=version)
    else:
        # 还没提交：删掉写了一半的这个版本，已提交的版本原样保留
        store.delete_document(doc_id, only_version=version)
    log.warning("收尾上次中断的入库：doc=%s version=%d", doc_id, version)


def mark(db: Database, store: MilvusStore, doc_id: str, version: int) -> None:
    raw = db.get_meta(_key(doc_id))
    if raw:
        entry = json.loads(raw)
        if _in_flight(entry):
            raise IngestInProgress(f"文档 {doc_id} 正在被另一个进程（pid {entry['pid']}）入库，等它结束后再试")
        # 上一次没收尾（删旧版本失败、清理失败）：先按那次的版本收尾，再覆盖成这一次
        _settle(db, store, doc_id, int(entry["version"]))
    db.set_meta(_key(doc_id), json.dumps({"version": version, "pid": os.getpid(), "at": time.time()}))


def clear(db: Database, doc_id: str) -> None:
    db.delete_meta(_key(doc_id))


def recover(db: Database, store: MilvusStore) -> int:
    settled = []
    for doc_id, entry in _entries(db).items():
        if _in_flight(entry):
            log.info("doc=%s 正由 pid %s 入库，跳过收尾", doc_id, entry.get("pid"))
            continue
        _settle(db, store, doc_id, int(entry["version"]))
        settled.append(doc_id)
    if settled:
        # 先加版本号再清记录：中间被杀时记录还在，下次启动重做一遍（每一步都能重复执行）。
        # 版本号加一让缓存失效；ES 的同步版本随之落后，启动时 sync_es 会按 Milvus 重建
        db.bump_kb_version()
        for doc_id in settled:
            clear(db, doc_id)
    return len(settled)

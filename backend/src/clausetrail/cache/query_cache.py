import hashlib
import json
import logging
import re
import time
import unicodedata
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Protocol

from clausetrail import lexicon
from clausetrail.answer.prompt import SYSTEM_PROMPT
from clausetrail.breaker import Breaker
from clausetrail.config import Settings, redact_url

log = logging.getLogger(__name__)

KEY_PREFIX = "clausetrail:qa:"
MEMORY_CAPACITY = 512


# 转换器加载词典较慢，只建一次；不可用时提示一次，缓存 key 不做繁简归一
@lru_cache(maxsize=1)
def _t2s():
    try:
        import opencc

        return opencc.OpenCC("t2s")
    except Exception:
        log.warning("OpenCC 不可用，缓存 key 不做繁简归一", exc_info=True)
        return None


def _to_simplified(text: str) -> str:
    converter = _t2s()
    if converter is None:
        return text
    try:
        return converter.convert(text)
    except Exception:
        return text


PUNCT_OR_SPACE = re.compile(r"[\W_]+", re.UNICODE)
# 小数点和百分号是条款问题里的关键信息（3.5% 和 35 不是同一个问题），去标点前先换成文字
DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")
# 数字之间的区间号、斜杠同理：「第1-3年」和「第13年」、「1/2」和「12」不是同一个问题
NUMBER_RANGE = re.compile(r"(?<=\d)\s*[-~—–]\s*(?=\d)")
NUMBER_SLASH = re.compile(r"(?<=\d)\s*/\s*(?=\d)")
FILLERS = ("请问", "一下", "吗", "呢", "吧", "啊", "的", "了")


@lru_cache(maxsize=1)
def _alias_pattern() -> tuple[re.Pattern | None, dict[str, str]]:
    table = lexicon.aliases()
    keys = set(table) | set(table.values())
    if not keys:
        return None, table
    # 按长度倒序拼正则，让长术语先匹配，不被它包含的短别名截断
    ordered = sorted(keys, key=len, reverse=True)
    return re.compile("|".join(re.escape(k) for k in ordered)), table


# 抹平繁简、全半角（NFKC）、别名、标点和语气词的差异，说法不同的同一个问题落到同一个缓存 key
def normalize_query(question: str) -> str:
    text = unicodedata.normalize("NFKC", _to_simplified(question)).lower()
    pattern, table = _alias_pattern()
    if pattern is not None:
        text = pattern.sub(lambda m: table.get(m.group(0), m.group(0)), text)
    text = DECIMAL_POINT.sub("点", text).replace("%", "pct")
    text = NUMBER_SLASH.sub("比", NUMBER_RANGE.sub("至", text))
    text = PUNCT_OR_SPACE.sub("", text)
    for filler in FILLERS:
        text = text.replace(filler, "")
    return text


# key 带上知识库版本和配置指纹：入库、删文档、换库、改模型 / 提示词 / 检索参数 / 词表后旧缓存自然失效
def cache_key(standalone_query: str, kb_version: int, fingerprint: str) -> str:
    raw = f"{normalize_query(standalone_query)}|kb={kb_version}|cfg={fingerprint}"
    return KEY_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()


# kb_epoch 是建库时生成的随机值：删掉 SQLite 重建后版本号从 0 重来，不会撞上 Redis 里的旧 key
def config_fingerprint(settings: Settings, kb_epoch: str) -> str:
    parts = [
        kb_epoch,
        lexicon.fingerprint(),
        settings.llm_answer_model,
        settings.embedder_backend,
        settings.embedding_model,
        settings.reranker_backend,
        settings.reranker_model,
        settings.aliyun_opensearch_ranker_id,
        str(settings.recall_top_k),
        str(settings.rrf_k),
        str(settings.rerank_top_k),
        str(settings.context_top_k),
        str(settings.refusal_threshold),
        SYSTEM_PROMPT,
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


class QueryCache(Protocol):
    name: str

    async def get(self, key: str) -> dict[str, Any] | None: ...

    async def set(self, key: str, value: dict[str, Any]) -> None: ...


class RedisCache:
    name = "redis"

    def __init__(self, url: str, ttl: int, degraded: dict[str, str] | None = None):
        import redis.asyncio as aioredis
        from redis.exceptions import ConnectionError as RedisConnectionError
        from redis.exceptions import TimeoutError as RedisTimeoutError

        # 主机不可达（丢包而不是拒绝连接）时，不能让每次读写缓存都拖到系统超时
        self._redis = aioredis.from_url(url, decode_responses=True, socket_connect_timeout=1, socket_timeout=2)
        # 只有连不上、超时才熔断；别的异常（如存进去的值不是合法 JSON）照常抛给调用方，按未命中处理
        self._outages = (RedisConnectionError, RedisTimeoutError, OSError)
        self._sync_url = url
        self._ttl = ttl
        # 运行中 Redis 出故障时，冷却期内改用进程内 LRU，不让每次读写都先等超时
        self._fallback = MemoryCache(ttl)
        self._breaker = Breaker("cache", degraded if degraded is not None else {})
        self._reason = f"Redis 读写失败（{redact_url(url)}），暂时改用进程内 LRU"

    # build_cache 在同步的启动流程里调用，所以用同步客户端 ping；连接和读取各最多等 1 秒，用完就关
    def available(self) -> bool:
        import redis

        client = redis.Redis.from_url(self._sync_url, socket_connect_timeout=1, socket_timeout=1)
        try:
            return bool(client.ping())
        except Exception:
            return False
        finally:
            client.close()

    def _trip(self) -> None:
        log.warning(self._reason, exc_info=True)
        self._breaker.failure(self._reason)

    async def get(self, key: str) -> dict[str, Any] | None:
        if self._breaker.is_open():
            return await self._fallback.get(key)
        try:
            raw = await self._redis.get(key)
        except self._outages:
            self._trip()
            return await self._fallback.get(key)
        self._breaker.success()
        return json.loads(raw) if raw else None

    async def set(self, key: str, value: dict[str, Any]) -> None:
        if self._breaker.is_open():
            await self._fallback.set(key, value)
            return
        payload = json.dumps(value, ensure_ascii=False)
        try:
            await self._redis.set(key, payload, ex=self._ttl)
        except self._outages:
            self._trip()
            await self._fallback.set(key, value)
            return
        self._breaker.success()

    async def aclose(self) -> None:
        close = getattr(self._redis, "aclose", None) or self._redis.close
        await close()


class MemoryCache:
    name = "memory"

    def __init__(self, ttl: int, capacity: int = MEMORY_CAPACITY):
        self._ttl = ttl
        self._capacity = capacity
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    async def get(self, key: str) -> dict[str, Any] | None:
        item = self._items.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.monotonic():
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return value

    async def set(self, key: str, value: dict[str, Any]) -> None:
        self._items[key] = (time.monotonic() + self._ttl, value)
        self._items.move_to_end(key)
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)


def build_cache(settings: Settings, degraded: dict[str, str] | None = None) -> tuple[QueryCache, str | None]:
    if settings.redis_url:
        cache = RedisCache(settings.redis_url, settings.cache_ttl_seconds, degraded)
        if cache.available():
            return cache, None
        reason = f"连不上 Redis（{redact_url(settings.redis_url)}），缓存降级为进程内 LRU"
    else:
        reason = "未配置 REDIS_URL，缓存使用进程内 LRU"
    log.warning(reason)
    return MemoryCache(settings.cache_ttl_seconds), reason

import asyncio
import time

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from clausetrail import lexicon
from clausetrail.answer.citations import CitationFilter, normalize_markers
from clausetrail.breaker import Breaker
from clausetrail.cache.query_cache import RedisCache, normalize_query
from clausetrail.chunking.text_utils import count_tokens
from clausetrail.index.bm25 import ES_LEXICON_KEY, sync_es
from clausetrail.ingest.native_pdf import _is_bold, drop_superscripts
from clausetrail.models.reranker import TRUNCATION_MARGIN, truncate_for_pair
from clausetrail.storage.db import Database
from clausetrail.structure.ocr_correction import correct_text, load_confusions, load_terms


def test_number_ranges_do_not_collide_in_cache_key():
    assert normalize_query("保单第1-3年的现金价值") != normalize_query("保单第13年的现金价值")
    assert normalize_query("1/2") != normalize_query("12")
    assert normalize_query("60~65岁") == normalize_query("60-65岁")


def test_loose_citation_markers_are_normalized():
    assert normalize_markers("见[1,3]和【2】") == "见[1][3]和[2]"
    assert normalize_markers("【重要提示】请阅读") == "【重要提示】请阅读"
    f = CitationFilter(3)
    out = f.feed("等待期为90日［1") + f.feed("，3］。") + f.flush()
    assert out == "等待期为90日[1][3]。"


def test_alias_expansion_uses_indexed_word_forms():
    tokens = lexicon.query_tokens("癌症的等待期").split()
    assert "恶性肿瘤" in tokens
    assert not any("——" in t for t in tokens)


def test_truncation_keeps_longest_prefix_within_budget():
    query = "等待期多久"
    doc = "被保险人在等待期内确诊重大疾病的，我们无息退还已交保险费。" * 40
    out = truncate_for_pair(query, doc, 120)
    budget = 120 - count_tokens(query) - TRUNCATION_MARGIN
    assert count_tokens(out) <= budget < count_tokens(doc[: len(out) + 1])


def test_breaker_opens_then_recovers():
    degraded: dict[str, str] = {}
    b = Breaker("bm25", degraded, cooldown=0.05)
    b.failure("es down")
    assert b.is_open() and degraded == {"bm25": "es down"}
    time.sleep(0.06)
    assert not b.is_open()
    b.success()
    assert degraded == {}


def test_bold_detection_ignores_regular_weights():
    assert _is_bold("XEKKFX+SimHei") and _is_bold("Arial-BoldMT") and _is_bold("Times New Roman,Bold")
    assert not _is_bold("MicrosoftYaHei") and not _is_bold("Helvetica-Book") and not _is_bold("LNUHNF+SimSun")
    assert not _is_bold("STHeitiSC-Light") and not _is_bold("MicrosoftJhengHeiLight")


class _BrokenRedis:
    def __init__(self, exc: Exception):
        self.exc = exc

    async def get(self, key):
        raise self.exc

    async def set(self, key, value, ex=None):
        raise self.exc


def test_redis_breaker_trips_only_on_connection_errors():
    degraded: dict[str, str] = {}
    cache = RedisCache("redis://127.0.0.1:1/0", 60, degraded)
    cache._redis = _BrokenRedis(RedisConnectionError("down"))
    assert asyncio.run(cache.get("k")) is None
    assert "cache" in degraded
    # 冷却期内读写都走进程内 LRU
    asyncio.run(cache.set("k", {"a": 1}))
    assert asyncio.run(cache.get("k")) == {"a": 1}
    other = RedisCache("redis://127.0.0.1:1/0", 60, {})
    other._redis = _BrokenRedis(ValueError("bad value"))
    with pytest.raises(ValueError):
        asyncio.run(other.get("k"))
    assert not other._breaker.is_open()


class _FakeES:
    def __init__(self):
        self.rebuilds = 0

    def has_index(self) -> bool:
        return True

    def rebuild(self, rows) -> None:
        self.rebuilds += 1


def test_es_index_rebuilds_when_lexicon_changes(tmp_path):
    db = Database(tmp_path / "t.sqlite3")
    es = _FakeES()
    sync_es(es, db, None)
    sync_es(es, db, None)
    assert es.rebuilds == 1
    # 词表改过（指纹对不上）：同义词写在分析器里，要整个重建
    db.set_meta(ES_LEXICON_KEY, "old")
    sync_es(es, db, None)
    assert es.rebuilds == 2
    db.close()


def test_short_term_correction_skips_real_words():
    assert correct_text("我们给什么保障", load_terms(), load_confusions())[1] == []


class _FakePage:
    """只提供 drop_superscripts 用到的 chars 和 filter。"""

    def __init__(self, chars: list[dict]):
        self.chars = chars

    def filter(self, keep):
        return _FakePage([c for c in self.chars if keep(c)])


def _char(text: str, size: float, top: float, x: float = 0.0) -> dict:
    return {
        "object_type": "char",
        "text": text,
        "size": size,
        "top": top,
        "bottom": top + size,
        "x0": x,
        "x1": x + size,
    }


def _line(parts: list[tuple[str, float, float]]) -> list[dict]:
    chars, x = [], 0.0
    for text, size, top in parts:
        chars.append(_char(text, size, top, x))
        x += size
    return chars


def test_small_print_rows_keep_their_digits():
    body = [_char("保", 10, 100, 10 * i) for i in range(10)]
    mark = _char("5", 5, 99, 100)
    row = [_char(t, 5, 300, 5 * i) for i, t in enumerate("100%")]
    kept = drop_superscripts(_FakePage([*body, mark, *row])).chars
    assert mark not in kept
    assert all(c in kept for c in row)


def test_formula_subscripts_are_kept_but_footnote_marks_dropped():
    # 「T₁N₀M₀期或更轻分期⁵的」：跟在字母后面的下标是分期的一部分，跟在汉字后面的 5 是脚注角标
    parts = [("T", 10, 100), ("1", 5, 105), ("N", 10, 100), ("0", 5, 105), ("M", 10, 100), ("0", 5, 105)]
    parts += [(t, 10, 100) for t in "期或更轻分期"] + [("5", 5, 99), ("的", 10, 100)]
    kept = drop_superscripts(_FakePage(_line(parts))).chars
    assert "".join(c["text"] for c in kept) == "T1N0M0期或更轻分期的"
    # 连着的一串上标按这串前面的字判断：「（1＋年利率）」后面的 1/360 整串保留
    parts = [(t, 10, 100) for t in "日利率＝（1＋年利率）"] + [(t, 5, 99) for t in "1/360"] + [("－", 10, 100)]
    assert "".join(c["text"] for c in drop_superscripts(_FakePage(_line(parts))).chars).endswith("）1/360－")

import csv
import hashlib
import logging
import re
from functools import lru_cache

import jieba

from clausetrail.config import get_settings

log = logging.getLogger(__name__)
jieba.setLogLevel(logging.WARNING)

# 词频给得很高，保证 jieba 把术语当整词切，不被通用词典拆开
USER_WORD_FREQ = 20000
STOP_TOKENS = frozenset(
    [
        "的",
        "了",
        "和",
        "与",
        "及",
        "或",
        "是",
        "在",
        "对",
        "就",
        "也",
        "都",
        "而",
        "被",
        "把",
        "吗",
        "呢",
        "吧",
        "啊",
        "么",
        "什么",
        "怎么",
        "如何",
        "哪些",
        "是否",
        "请问",
    ]
)
PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, str], frozenset[str]]:
    terms_dir = get_settings().resources_dir / "terms"
    alias_to_canonical: dict[str, str] = {}
    words: set[str] = set()
    terms_csv = terms_dir / "insurance_terms.csv"
    if terms_csv.exists():
        with terms_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                term = row["term"].strip()
                words.add(term)
                for alias in filter(None, (row.get("aliases") or "").split("|")):
                    alias_to_canonical[alias.strip()] = term
                    words.add(alias.strip())
    synonyms_csv = terms_dir / "synonyms.csv"
    if synonyms_csv.exists():
        with synonyms_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                alias_to_canonical[row["alias"].strip()] = row["canonical"].strip()
                words.update((row["alias"].strip(), row["canonical"].strip()))
    for w in words:
        if len(w) >= 2:
            jieba.add_word(w, freq=USER_WORD_FREQ)
    log.debug("lexicon loaded: %d words, %d aliases", len(words), len(alias_to_canonical))
    return alias_to_canonical, frozenset(words)


def ensure_loaded() -> None:
    _load()


def aliases() -> dict[str, str]:
    return _load()[0]


def vocabulary() -> frozenset[str]:
    return _load()[1]


# 术语表、同义词表的内容指纹，放进缓存 key：改了词表，旧缓存不再命中
@lru_cache(maxsize=1)
def fingerprint() -> str:
    terms_dir = get_settings().resources_dir / "terms"
    digest = hashlib.sha256()
    for name in ("insurance_terms.csv", "synonyms.csv"):
        p = terms_dir / name
        if p.exists():
            digest.update(p.read_bytes())
    return digest.hexdigest()[:12]


# 术语在索引里实际的样子。jieba 切不出「——」这类符号，「恶性肿瘤——重度」在索引里是「恶性肿瘤」「重度」两个词；
# 查询扩展和 ES 同义词都要用这个形式，否则永远对不上。
# 同一个术语每次查询都会用到，结果缓存起来；返回元组，防调用方改动缓存里的值
@lru_cache(maxsize=4096)
def index_form(term: str) -> tuple[str, ...]:
    ensure_loaded()
    words = tuple(t for t in jieba.lcut(term) if t.strip() and not PUNCT.match(t))
    return words or (term,)


def tokenize(text: str) -> list[str]:
    ensure_loaded()
    # 搜索引擎模式：长词里在词典中的二字、三字子词也切出来，提高 BM25 召回
    return [t for t in jieba.lcut_for_search(text) if t.strip() and not PUNCT.match(t)]


def canonicalize_tokens(tokens: list[str]) -> list[str]:
    table = aliases()
    return [table.get(t, t) for t in tokens]


def index_tokens(text: str) -> str:
    return " ".join(tokenize(text))


def query_tokens(text: str) -> str:
    tokens = [t for t in tokenize(text) if t not in STOP_TOKENS]
    table = aliases()
    # 命中别名时追加标准术语；Milvus BM25 没有同义词过滤器，靠这一步对齐别名和标准术语
    expanded = list(tokens)
    for t in tokens:
        canonical = table.get(t)
        if not canonical:
            continue
        for w in index_form(canonical):
            if w not in expanded:
                expanded.append(w)
    return " ".join(expanded)

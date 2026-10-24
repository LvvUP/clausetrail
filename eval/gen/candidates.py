# 从切好的片段里挑 gold 句子：术语的定义句、章节的开头几句
import csv
import re
from collections import Counter
from dataclasses import dataclass

from clausetrail import lexicon
from clausetrail.chunking.splitter import is_toc_like
from clausetrail.chunking.text_utils import split_sentences
from clausetrail.config import get_settings
from clausetrail.types import Chunk
from eval.gen.corpus import Doc
from eval.metrics import norm

MIN_SPAN, MAX_SPAN = 12, 160
CJK = re.compile(r"[一-鿿]")
NUMBERING = re.compile(
    r"^\s*(第[一二三四五六七八九十百零〇]+[章节条款]|\d{1,3}(\.\d{1,2})*[.、．;；]?|[（(][一二三四五六七八九十\d]{1,3}[）)]|[一二三四五六七八九十]+、)\s*"
)
GENERIC_TITLES = {
    "阅读指引",
    "条款目录",
    "目录",
    "释义",
    "附则",
    "说明",
    "特别提示",
    "您拥有的重要权益",
    "您应当特别注意的事项",
}
TERM_CATEGORIES = {
    "期间",
    "保险费",
    "保险金与保额",
    "合同构成与效力",
    "责任免除",
    "理赔",
    "健康险与医疗",
    "寿险与年金",
    "意外与伤残",
    "合同主体",
    "重疾病种",
}
# 改写集只用「属性类」术语，病种名不适合「那 XX 呢」这种追问
REWRITE_CATEGORIES = TERM_CATEGORIES - {"重疾病种"}
EXCLUSION_KEYS = ("责任免除", "免责")
# 在一份文档超过四分之一的章节里都出现的术语（如「被保险人」）太泛，不用来出细节题
GENERIC_SHARE = 0.25
DETAIL_PER_SECTION = 3


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    chunk: Chunk
    span: str
    kind: str  # term / section / detail
    key: str  # 标准术语，或清洗后的章节标题
    title: str = ""  # 细节题所在章节的标题
    surface: str = ""  # 原文里实际出现的写法，可能是别名


def own_text(chunk: Chunk) -> str:
    """去掉片段开头与上一片段重叠的那句。"""
    n = chunk.char_end - chunk.char_start
    return chunk.text[-n:] if 0 < n <= len(chunk.text) else chunk.text


def spans(chunk: Chunk) -> list[str]:
    return [
        s
        for s in split_sentences(own_text(chunk))
        if MIN_SPAN <= len(s) <= MAX_SPAN and CJK.search(s) and not is_toc_like(s)
    ]


def clean_title(title: str) -> str:
    return NUMBERING.sub("", title).strip(" *：:　")


def terms_in(categories: set[str]) -> list[str]:
    path = get_settings().resources_dir / "terms" / "insurance_terms.csv"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return sorted(
        {r["term"].strip() for r in rows if r["category"].strip() in categories and len(r["term"].strip()) >= 2}
    )


def aliases_by_term() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for alias, term in lexicon.aliases().items():
        if alias != term and len(alias) >= 2:
            out.setdefault(term, []).append(alias)
    return {t: sorted(a) for t, a in out.items()}


def surface_map(categories: set[str]) -> dict[str, str]:
    """原文写法 → 标准术语。条款里常写别名（如「观察期」），出题和判重都按标准术语算。"""
    terms = terms_in(categories)
    out = {t: t for t in terms}
    for alias, canon in lexicon.aliases().items():
        if canon in out and len(alias) >= 2:
            out[alias] = canon
    return out


def _definition(text: str, term: str) -> str | None:
    # 「犹豫期指…」「冠状动脉搭桥术（或称…）*; 指…」：术语和「指」之间只允许别名括注、星号和标点；
    # 前后紧挨着汉字的不算（「身故保险金指」之于「保险金」，「护理费：指」之于「护理」）
    m = re.search(
        rf"(?<![一-鿿]){re.escape(term)}(?:[（(][^）)]{{0,20}}[）)])?[*＊]?[\s：:；;，,]*(是指|系指|指(?![定南导示标出明]))",
        text,
    )
    if not m:
        return None
    end = text.find("。", m.end())
    span = text[m.start() : end + 1 if end >= 0 else len(text)][:MAX_SPAN].strip()
    return span if len(span) >= MIN_SPAN else None


def _first_chunks(doc: Doc) -> dict[tuple[str, ...], Chunk]:
    first: dict[tuple[str, ...], Chunk] = {}
    for c in doc.chunks:
        first.setdefault(tuple(c.section_path), c)
    return first


def term_candidates(doc: Doc, surfaces: dict[str, str]) -> list[Candidate]:
    """每个标准术语在每份文档里最多一个候选：章节标题就是该术语（或别名）的优先，其次是定义句。
    两个术语对到同一句原文时只保留先找到的那个，一句 gold 只出一道题。"""
    found: dict[str, Candidate] = {}
    taken: set[str] = set()

    def add(chunk: Chunk, span: str, canon: str, surface: str) -> None:
        if canon not in found and norm(span) not in taken:
            found[canon] = Candidate(doc.doc_id, chunk, span, "term", canon, surface=surface)
            taken.add(norm(span))

    for path, chunk in _first_chunks(doc).items():
        title = clean_title(path[-1]) if path else ""
        if (canon := surfaces.get(title)) and (ss := spans(chunk)):
            add(chunk, ss[0], canon, title)
    for chunk in doc.chunks:
        text = own_text(chunk)
        for surface, canon in surfaces.items():
            if canon not in found and surface in text and (span := _definition(text, surface)):
                add(chunk, span, canon, surface)
    return [found[t] for t in sorted(found)]


def valid_title(doc: Doc, title: str) -> bool:
    return (
        2 <= len(title) <= 20
        and bool(CJK.search(title))
        and title not in GENERIC_TITLES
        and not title.endswith("条款")
        and "有限公司" not in title
        and title not in (doc.product, doc.short)
    )


def section_candidates(doc: Doc, used: set[str], per_section: int = 2) -> list[list[Candidate]]:
    """每个章节一道题：首个片段里还没被用作 gold 的前两句都算 gold。
    used 存去掉空白后的句子，选中的句子会加进去；不同章节里重复出现的标准文本只用一次。"""
    out = []
    for path, chunk in _first_chunks(doc).items():
        if not path or not valid_title(doc, title := clean_title(path[-1])):
            continue
        picked: list[str] = []
        for s in spans(chunk):
            if len(picked) < per_section and norm(s) not in used:
                picked.append(s)
                used.add(norm(s))
        if picked:
            out.append([Candidate(doc.doc_id, chunk, s, "section", title) for s in picked])
    return out


def _matches(sentence: str, by_len: list[str], surfaces: dict[str, str]) -> set[str]:
    """句子里出现的标准术语；长写法优先，被长写法包含的短写法不算（「身故保险金」里不再算「保险金」）。"""
    taken: list[str] = []
    for s in by_len:
        if s in sentence and not any(s in t for t in taken):
            taken.append(s)
    return {surfaces[s] for s in taken}


def detail_candidates(
    doc: Doc, surfaces: dict[str, str], used: set[str], per_section: int = DETAIL_PER_SECTION
) -> list[Candidate]:
    """章节里的具体规定：某个术语在这一节里只出现在一句话中，就以「这一节的这个术语」出题，gold 是这句话。
    会把选中的句子加进 used，保证一句 gold 只对应一道题。"""
    sections: dict[tuple[str, ...], list[Chunk]] = {}
    for c in doc.chunks:
        if c.section_path and valid_title(doc, clean_title(c.section_path[-1])):
            sections.setdefault(tuple(c.section_path), []).append(c)
    by_len = sorted(surfaces, key=len, reverse=True)
    hits = {p: [(c, s, _matches(s, by_len, surfaces)) for c in cs for s in spans(c)] for p, cs in sections.items()}
    spread = Counter(k for hs in hits.values() for k in set().union(*(ks for _, _, ks in hs)))
    limit = max(2.0, GENERIC_SHARE * len(sections))
    out = []
    for path, hs in hits.items():
        title = clean_title(path[-1])
        per_term = Counter(k for _, _, ks in hs for k in ks)
        picked = 0
        for chunk, sentence, ks in hs:
            if picked >= per_section or norm(sentence) in used:
                continue
            for k in sorted(ks):
                if (
                    per_term[k] == 1
                    and spread[k] <= limit
                    and k not in title
                    and title not in k
                    and k not in doc.product
                ):
                    out.append(Candidate(doc.doc_id, chunk, sentence, "detail", k, title=title))
                    used.add(norm(sentence))
                    picked += 1
                    break
    return out

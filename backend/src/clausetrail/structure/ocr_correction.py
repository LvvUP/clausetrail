import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jieba

from clausetrail.config import get_settings
from clausetrail.types import Block, ParserName

# 只纠正 OCR 置信度低于这个值的行，高置信度的行不动，减少误改
LOW_CONFIDENCE = 0.92
OCR_PARSERS = {ParserName.PADDLEOCR, ParserName.DEEPDOC_OCR}


@dataclass(frozen=True)
class Correction:
    page: int
    wrong: str
    right: str


@lru_cache(maxsize=1)
def load_terms(path: str | None = None) -> tuple[str, ...]:
    p = Path(path) if path else get_settings().resources_dir / "terms" / "insurance_terms.csv"
    if not p.exists():
        return ()
    with p.open(encoding="utf-8") as f:
        terms = {row["term"].strip() for row in csv.DictReader(f) if row.get("term")}
    # 长词排在前面，correct_text 按顺序匹配时优先用完整的长术语
    return tuple(sorted(terms, key=len, reverse=True))


@lru_cache(maxsize=1)
def load_confusions(path: str | None = None) -> dict[str, frozenset[str]]:
    p = Path(path) if path else get_settings().resources_dir / "terms" / "ocr_confusions.csv"
    table: dict[str, set[str]] = {}
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                a, b = row["char"].strip(), row["confusable"].strip()
                table.setdefault(a, set()).add(b)
                table.setdefault(b, set()).add(a)
    return {k: frozenset(v) for k, v in table.items()}


# 只接受混淆字表里的替换，且替换字数有上限（至少 1 个，约每 4 个字 1 个），免得把差异很大的片段硬改成术语
def _matches_with_confusion(window: str, term: str, confusions: dict[str, frozenset[str]]) -> bool:
    diffs = 0
    for got, want in zip(window, term, strict=True):
        if got == want:
            continue
        if got not in confusions.get(want, frozenset()):
            return False
        diffs += 1
        if diffs > max(1, len(term) // 4):
            return False
    return diffs > 0


SHORT_TERM_CHARS = 2


# 两字术语只差一个字时很容易撞上正常文字：「给什么」里的「给什」会被改成「给付」。
# 被替换的字和窗口外紧挨着的字能组成常用词（如「什么」）时，说明这里本来就不是这个术语，不改
def _breaks_word(chars: list[str], i: int, window: str, term: str) -> bool:
    if len(term) > SHORT_TERM_CHARS:
        return False
    jieba.dt.check_initialized()
    freq = jieba.dt.FREQ
    end = i + len(term)
    for k, (got, want) in enumerate(zip(window, term, strict=True)):
        if got == want:
            continue
        if k == 0 and i > 0 and freq.get(chars[i - 1] + got, 0) > 0:
            return True
        if k == len(term) - 1 and end < len(chars) and freq.get(got + chars[end], 0) > 0:
            return True
    return False


def correct_text(
    text: str, terms: tuple[str, ...], confusions: dict[str, frozenset[str]]
) -> tuple[str, list[tuple[str, str]]]:
    fixes: list[tuple[str, str]] = []
    chars = list(text)
    i = 0
    while i < len(chars):
        matched = False
        for term in terms:
            n = len(term)
            if n < 2 or i + n > len(chars):
                continue
            window = "".join(chars[i : i + n])
            # 已经正确的术语整段跳过，免得它的后半段和别的术语重叠时被误改
            if window == term:
                i += n
                matched = True
                break
            if _matches_with_confusion(window, term, confusions) and not _breaks_word(chars, i, window, term):
                fixes.append((window, term))
                chars[i : i + n] = list(term)
                i += n
                matched = True
                break
        if not matched:
            i += 1
    return "".join(chars), fixes


def correct_blocks(blocks: list[Block]) -> list[Correction]:
    terms, confusions = load_terms(), load_confusions()
    if not terms or not confusions:
        return []
    applied: list[Correction] = []
    for b in blocks:
        if b.parser not in OCR_PARSERS or b.ocr_confidence >= LOW_CONFIDENCE:
            continue
        fixed, fixes = correct_text(b.text, terms, confusions)
        if fixes:
            b.text = fixed
            applied.extend(Correction(b.page, w, r) for w, r in fixes)
    return applied

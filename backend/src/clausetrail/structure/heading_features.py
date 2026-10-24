import re
import statistics
from collections import Counter

import numpy as np

from clausetrail.types import Block, BlockKind

LEVEL_BODY, LEVEL_CHAPTER, LEVEL_ARTICLE, LEVEL_ITEM = 0, 1, 2, 3

NUMBERING_PATTERNS: list[tuple[str, re.Pattern, int]] = [
    ("chapter_cn", re.compile(r"^第[一二三四五六七八九十百零〇]+[章部分]"), LEVEL_CHAPTER),
    ("article_cn", re.compile(r"^第[一二三四五六七八九十百零〇]+条"), LEVEL_ARTICLE),
    ("decimal_1", re.compile(r"^\d{1,2}(?:\s|\.(?!\d)|、)\s*[^\d\s]"), LEVEL_CHAPTER),
    ("decimal_2", re.compile(r"^\d{1,2}\.\d{1,2}(?!\.\d)\s*[^\d\s.]"), LEVEL_ARTICLE),
    ("decimal_3", re.compile(r"^\d{1,2}\.\d{1,2}\.\d{1,2}\s*[^\d\s.]"), LEVEL_ITEM),
    ("cn_dun", re.compile(r"^[一二三四五六七八九十]+、"), LEVEL_ARTICLE),
    ("paren_cn", re.compile(r"^[（(][一二三四五六七八九十]+[）)]"), LEVEL_ITEM),
    ("paren_num", re.compile(r"^[（(]\d{1,2}[）)]"), LEVEL_ITEM),
]

SENTENCE_END = re.compile(r"[。；;！!？?]$")
LEAD_IN_END = re.compile(r"[:：]$")
# 公式行（「金额 = … × 给付系数」）常居中、字号大，但不是标题
FORMULA = re.compile(r"[=＝×÷]")
MIN_HEADING_CHARS = 2
TOC_DOTS = re.compile(r"[.…·]{3,}\s*\d+$")
# 行尾顶到版心右边界，说明是折行的正文，不是标题
WRAPPED_REACH = 0.95
# 不加粗、只比正文大一号的短行也算标题（平安条款的章标题就是这样排的）
LARGER_FONT_RATIO = 1.1

FEATURE_NAMES = [
    "rel_font_size",
    "is_bold",
    "indent_ratio",
    "width_ratio",
    "right_reach",
    "length",
    "ends_with_punct",
    "ends_with_colon",
    "formula_like",
    "layout_title",
    "gap_above",
    "gap_below",
    "first_on_page",
    "toc_like",
    *[f"num_{name}" for name, _, _ in NUMBERING_PATTERNS],
]
_COL = {name: i for i, name in enumerate(FEATURE_NAMES)}


def numbering(text: str) -> tuple[str | None, int]:
    head = text.strip()
    for name, pattern, level in NUMBERING_PATTERNS:
        if pattern.match(head):
            return name, level
    return None, LEVEL_BODY


def _page_stats(blocks: list[Block]) -> dict[int, tuple[float, float, float]]:
    stats: dict[int, tuple[float, float, float]] = {}
    by_page: dict[int, list[Block]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)
    for page, items in by_page.items():
        # 按字数加权取正文字号，脚注多的页面也不会把脚注字号当成正文
        weights: Counter = Counter()
        for b in items:
            if b.font_size > 0:
                weights[round(b.font_size, 1)] += len(b.text)
        body_size = weights.most_common(1)[0][0] if weights else 10.0
        x0 = min(b.bbox[0] for b in items)
        x1 = max(b.bbox[2] for b in items)
        stats[page] = (body_size, x0, max(x1 - x0, 1.0))
    return stats


def extract_features(blocks: list[Block]) -> np.ndarray:
    stats = _page_stats(blocks)
    rows = []
    for i, b in enumerate(blocks):
        body_size, left, width = stats[b.page]
        prev = blocks[i - 1] if i > 0 and blocks[i - 1].page == b.page else None
        nxt = blocks[i + 1] if i + 1 < len(blocks) and blocks[i + 1].page == b.page else None
        line_h = max(b.bbox[3] - b.bbox[1], 1.0)
        text = b.text.strip()
        num_name, _ = numbering(text)
        row = [
            b.font_size / body_size if body_size else 1.0,
            float(b.bold),
            (b.bbox[0] - left) / width,
            (b.bbox[2] - b.bbox[0]) / width,
            (b.bbox[2] - left) / width,
            min(len(text), 200) / 200,
            float(bool(SENTENCE_END.search(text))),
            float(bool(LEAD_IN_END.search(text))),
            float(bool(FORMULA.search(text))),
            float(b.kind == BlockKind.TITLE),
            ((b.bbox[1] - prev.bbox[3]) / line_h) if prev else 3.0,
            ((nxt.bbox[1] - b.bbox[3]) / line_h) if nxt else 3.0,
            float(prev is None),
            float(bool(TOC_DOTS.search(text))),
            *[float(num_name == name) for name, _, _ in NUMBERING_PATTERNS],
        ]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32).reshape(len(blocks), len(FEATURE_NAMES))


def rule_label(block: Block, rel_font_size: float, right_reach: float = 0.0) -> int:
    text = block.text.strip()
    if not text or block.kind in (BlockKind.TABLE, BlockKind.HEADER, BlockKind.FOOTER):
        return LEVEL_BODY
    if TOC_DOTS.search(text) or FORMULA.search(text) or len(text) < MIN_HEADING_CHARS:
        return LEVEL_BODY
    num_name, level = numbering(text)
    short = len(text) <= 40
    if num_name in ("chapter_cn", "article_cn") and len(text) <= 60:
        return level
    # 「1. 」「（1）」开头但折行到右边界、或以冒号引出下文的，是列表项/引导句
    if (
        num_name
        and short
        and not SENTENCE_END.search(text)
        and not LEAD_IN_END.search(text)
        and right_reach < WRAPPED_REACH
    ):
        return level
    if rel_font_size >= 1.35 and short:
        return LEVEL_CHAPTER
    standalone = (
        short and not SENTENCE_END.search(text) and not LEAD_IN_END.search(text) and right_reach < WRAPPED_REACH
    )
    emphasized = (block.bold or block.kind == BlockKind.TITLE) and rel_font_size >= 0.98
    if standalone and (emphasized or rel_font_size >= LARGER_FONT_RATIO):
        return LEVEL_ARTICLE if rel_font_size < 1.2 else LEVEL_CHAPTER
    return LEVEL_BODY


def rule_labels(blocks: list[Block]) -> list[int]:
    feats = extract_features(blocks)
    fs, reach = _COL["rel_font_size"], _COL["right_reach"]
    return [rule_label(b, float(feats[i, fs]), float(feats[i, reach])) for i, b in enumerate(blocks)]


def median_font(blocks: list[Block]) -> float:
    sizes = [b.font_size for b in blocks if b.font_size > 0]
    return statistics.median(sizes) if sizes else 10.0

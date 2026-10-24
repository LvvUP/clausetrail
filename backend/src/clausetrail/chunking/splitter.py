import hashlib
import re
from collections import Counter
from collections.abc import Callable

import numpy as np

from clausetrail.chunking.merge_rules import Unit, blocks_to_paragraphs, merge_by_similarity, merge_structural
from clausetrail.chunking.text_utils import count_tokens, split_sentences
from clausetrail.structure.tree import walk
from clausetrail.types import Chunk, SectionNode

Embedder = Callable[[list[str]], np.ndarray]

TOC_LINE = re.compile(r"^\s*(第[一二三四五六七八九十百零〇]+[章条]|\d+(\.\d+)*\s*\S)[^。；]{0,30}$")
ARTICLE_MARK = re.compile(r"第[一二三四五六七八九十百零〇]+[章条]")
OVERLAP_RESERVE_TOKENS = 80


def is_toc_like(text: str) -> bool:
    if "。" in text:
        return False
    marks = len(ARTICLE_MARK.findall(text))
    if marks >= 4 and marks * 12 >= len(text.replace("\n", "")) * 0.6:
        return True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    return sum(1 for ln in lines if TOC_LINE.match(ln)) / len(lines) >= 0.8


def _hard_wrap(sentence: str, max_tokens: int) -> list[str]:
    if count_tokens(sentence) <= max_tokens:
        return [sentence]
    pieces, buf = [], ""
    for ch in sentence:
        if buf and count_tokens(buf + ch) > max_tokens:
            pieces.append(buf)
            buf = ""
        buf += ch
    return [*pieces, buf] if buf else pieces


def _split_long(unit: Unit, max_tokens: int) -> list[Unit]:
    if unit.tokens <= max_tokens:
        return [unit]
    # 表格按行切，保留换行，避免把一行记录拆到两个片段里
    if unit.is_table:
        parts, sep = [w for row in unit.text.splitlines() if row.strip() for w in _hard_wrap(row, max_tokens)], "\n"
    else:
        parts, sep = [w for s in split_sentences(unit.text) for w in _hard_wrap(s, max_tokens)], ""
    pieces, buf = [], ""
    for part in parts:
        if buf and count_tokens(buf + sep + part) > max_tokens:
            pieces.append(buf)
            buf = ""
        buf = f"{buf}{sep}{part}" if buf else part
    if buf:
        pieces.append(buf)
    return _with_provenance(unit, pieces)


# 拆开后每一片按自己在原文里的位置找回覆盖到的块，页码和 bbox 只取这一片的，不沿用整个单元的范围
def _with_provenance(unit: Unit, pieces: list[str]) -> list[Unit]:
    spans, cursor = [], 0
    for b in unit.blocks:
        t = b.text.strip()
        pos = unit.text.find(t, cursor) if t else -1
        if pos >= 0:
            spans.append((pos, pos + len(t), b))
            cursor = pos + len(t)
    out, cursor = [], 0
    for p in pieces:
        # 句子切分可能去掉换行，整片不一定能原样找到，用开头一小段定位
        head = p[:16]
        start = unit.text.find(head, cursor)
        if start < 0:
            start = cursor
        end = start + len(p)
        # 游标推到这一片的末尾：条款里常有整句重复，只推过开头的话下一片会定位回同一处
        cursor = min(end, len(unit.text))
        blocks = [b for s, e, b in spans if s < end and e > start] or unit.blocks
        pages = [b.page for b in blocks] or [unit.page_start, unit.page_end]
        out.append(Unit(p, min(pages), max(pages), blocks, is_table=unit.is_table))
    return out


def _pack(units: list[Unit], max_tokens: int) -> list[Unit]:
    packed: list[Unit] = []
    for u in units:
        if packed and packed[-1].is_table == u.is_table and packed[-1].absorb(u).tokens <= max_tokens:
            packed[-1] = packed[-1].absorb(u)
        else:
            packed.append(u)
    return packed


def _node_text_tokens(node: SectionNode) -> int:
    own = sum(count_tokens(b.text) for b in node.blocks)
    return own + sum(_node_text_tokens(c) + count_tokens(c.title) for c in node.children)


def _collect_units(node: SectionNode) -> list[Unit]:
    units = blocks_to_paragraphs(node.blocks)
    for child in node.children:
        child_units = _collect_units(child)
        if child_units:
            head = Unit(child.title, child.page, child.page)
            child_units[0] = head.absorb(child_units[0])
            units.extend(child_units)
        else:
            units.append(Unit(child.title, child.page, child.page))
    return units


def section_units(root: SectionNode, max_tokens: int) -> list[tuple[list[str], list[Unit]]]:
    sections: list[tuple[list[str], list[Unit]]] = []

    def visit(node: SectionNode, ancestors: list[str]) -> None:
        path = node.path(ancestors)
        if node.title and _node_text_tokens(node) <= max_tokens:
            sections.append((path, _collect_units(node)))
            return
        own = blocks_to_paragraphs(node.blocks)
        if own:
            sections.append((path, own))
        for child in node.children:
            visit(child, path)

    visit(root, [])
    return sections


def _overlap_prefix(prev_text: str, n_sentences: int) -> str:
    if n_sentences <= 0:
        return ""
    sents = split_sentences(prev_text)
    return "".join(sents[-n_sentences:]) if len(sents) > n_sentences else ""


def chunk_document(
    root: SectionNode,
    doc_id: str,
    file_name: str,
    parser: str,
    max_tokens: int = 480,
    overlap_sentences: int = 1,
    embed: Embedder | None = None,
    similarity_threshold: float = 0.82,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    offset = 0
    budget = max_tokens - OVERLAP_RESERVE_TOKENS if overlap_sentences > 0 else max_tokens
    for path, units in section_units(root, max_tokens):
        units = merge_structural(units, max_tokens)
        units = merge_by_similarity(units, budget, similarity_threshold, embed)
        # 不超过上限的单元（整节、引导句 + 列表）保持完整，只是放不下重叠句；超过上限的才按预留重叠后的预算拆
        units = [piece for u in units for piece in (_split_long(u, budget) if u.tokens > max_tokens else [u])]
        units = _pack(units, budget)
        prev_text = ""
        for u in units:
            # 目录判断只看正文：以年龄、序号开头的短行表格（费率表、医院名单）格式上和目录很像，不能跳过
            if not u.is_table and is_toc_like(u.text):
                offset += len(u.text)
                continue
            overlap = "" if u.is_table else _overlap_prefix(prev_text, overlap_sentences)
            if overlap and count_tokens(overlap + u.text) > max_tokens:
                overlap = ""
            text = f"{overlap}{u.text}" if overlap else u.text
            chunk_id = hashlib.sha1(f"{doc_id}|{offset}|{u.text[:64]}".encode()).hexdigest()[:16]
            unit_parser = Counter(b.parser.value for b in u.blocks).most_common(1)[0][0] if u.blocks else parser
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    file_name=file_name,
                    text=text,
                    section_path=[p for p in path if p],
                    page_start=u.page_start,
                    page_end=u.page_end,
                    char_start=offset,
                    char_end=offset + len(u.text),
                    parser=unit_parser,
                    bboxes=[(b.page, *b.bbox) for b in u.blocks[:50]],
                )
            )
            offset += len(u.text)
            prev_text = u.text
    return chunks


def fixed_length_chunks(
    root: SectionNode, doc_id: str, file_name: str, size: int = 480, overlap: int = 50
) -> list[Chunk]:
    text = "\n".join(b.text for node, _ in walk(root) for b in node.blocks)
    step = max(1, size - overlap)
    chunks = []
    for i, start in enumerate(range(0, max(len(text), 1), step)):
        piece = text[start : start + size]
        if not piece.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}-fixed-{i}",
                doc_id=doc_id,
                file_name=file_name,
                text=piece,
                section_path=[],
                page_start=0,
                page_end=0,
                char_start=start,
                char_end=start + len(piece),
                parser="fixed",
            )
        )
    return chunks

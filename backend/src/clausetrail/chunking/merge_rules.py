from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from clausetrail.chunking.text_utils import count_tokens, is_list_item, is_list_lead_in, is_proviso
from clausetrail.types import Block, BlockKind


@dataclass
class Unit:
    text: str
    page_start: int
    page_end: int
    blocks: list[Block] = field(default_factory=list)
    is_table: bool = False

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)

    def absorb(self, other: "Unit", sep: str = "\n") -> "Unit":
        return Unit(
            text=f"{self.text}{sep}{other.text}",
            page_start=min(self.page_start, other.page_start),
            page_end=max(self.page_end, other.page_end),
            blocks=[*self.blocks, *other.blocks],
            is_table=self.is_table or other.is_table,
        )


def blocks_to_paragraphs(blocks: list[Block]) -> list[Unit]:
    paragraphs: list[Unit] = []
    current: Unit | None = None
    for b in blocks:
        text = b.text.strip()
        if not text:
            continue
        if b.kind == BlockKind.TABLE:
            if current:
                paragraphs.append(current)
                current = None
            paragraphs.append(Unit(text, b.page, b.page, [b], is_table=True))
            continue
        # 块是 PDF 的物理行：上一行没有以句末标点结束、本行也不是列表项或明显缩进时，按折行接到同一段
        starts_new = (
            current is None
            or current.text.endswith(("。", "；", "：", ":", "！", "？"))
            or is_list_item(text)
            or b.indent - (current.blocks[-1].indent if current.blocks else 0) > 12
        )
        if starts_new:
            if current:
                paragraphs.append(current)
            current = Unit(text, b.page, b.page, [b])
        else:
            current = current.absorb(Unit(text, b.page, b.page, [b]), sep="")
    if current:
        paragraphs.append(current)
    return paragraphs


def merge_structural(units: list[Unit], max_tokens: int) -> list[Unit]:
    merged: list[Unit] = []
    i = 0
    while i < len(units):
        u = units[i]
        # 冒号结尾的引导句把后面的列表项收进来，免得责任免除这类清单被拆散
        if is_list_lead_in(u.text):
            j = i + 1
            while j < len(units) and is_list_item(units[j].text) and u.absorb(units[j]).tokens <= max_tokens:
                u = u.absorb(units[j])
                j += 1
            merged.append(u)
            i = j
            continue
        # 「但……」「……除外」这类但书和条件句接到上一段，不和它限定的规定分开
        if merged and is_proviso(u.text) and merged[-1].absorb(u).tokens <= max_tokens:
            merged[-1] = merged[-1].absorb(u)
            i += 1
            continue
        merged.append(u)
        i += 1
    return merged


def merge_by_similarity(
    units: list[Unit],
    max_tokens: int,
    threshold: float,
    embed: Callable[[list[str]], np.ndarray] | None,
) -> list[Unit]:
    if embed is None or len(units) < 2:
        return units
    vecs = embed([u.text for u in units])
    vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
    out = [units[0]]
    last_vec = vecs[0]
    for u, v in zip(units[1:], vecs[1:], strict=True):
        if (
            not u.is_table
            and not out[-1].is_table
            and float(last_vec @ v) >= threshold
            and out[-1].absorb(u).tokens <= max_tokens
        ):
            out[-1] = out[-1].absorb(u)
            # 新段落和已合并内容的平均方向比，而不是只和最后一段比
            last_vec = (last_vec + v) / np.linalg.norm(last_vec + v)
        else:
            out.append(u)
            last_vec = v
    return out

from dataclasses import dataclass, field
from enum import StrEnum


class BlockKind(StrEnum):
    TEXT = "text"
    TITLE = "title"
    TABLE = "table"
    FIGURE = "figure"
    HEADER = "header"
    FOOTER = "footer"


class ParserName(StrEnum):
    DEEPDOC = "deepdoc"
    MINERU = "mineru"
    PADDLEOCR = "paddleocr"
    DEEPDOC_OCR = "deepdoc_ocr"
    DEEPDOC_TSR_FALLBACK = "deepdoc_tsr_fallback"
    DOCX = "docx"


@dataclass
class Block:
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    kind: BlockKind = BlockKind.TEXT
    font_size: float = 0.0
    bold: bool = False
    indent: float = 0.0
    parser: ParserName = ParserName.DEEPDOC
    level: int = 0
    ocr_confidence: float = 1.0


@dataclass
class SectionNode:
    title: str
    level: int
    page: int
    blocks: list[Block] = field(default_factory=list)
    children: list["SectionNode"] = field(default_factory=list)

    def path(self, ancestors: list[str]) -> list[str]:
        return [*ancestors, self.title] if self.title else ancestors


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    file_name: str
    text: str
    section_path: list[str]
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    parser: str
    kb_version: int = 0
    bboxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)

    @property
    def section_label(self) -> str:
        return " > ".join(p for p in self.section_path if p)

    # 片段正文里不一定出现所在章节的名称；向量、BM25、精排都用「章节标签 + 正文」
    @property
    def search_text(self) -> str:
        return f"{self.section_label}\n{self.text}" if self.section_label else self.text


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

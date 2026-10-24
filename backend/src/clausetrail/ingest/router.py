import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image, ImageOps, ImageSequence

from clausetrail.ingest import native_pdf, scanned, table
from clausetrail.types import Block, BlockKind

log = logging.getLogger(__name__)

SCAN_RENDER_DPI = 200
MIN_CHARS_PER_PAGE = 30
IMAGE_COVERAGE_FOR_SCAN = 0.8
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class PageType(StrEnum):
    NATIVE = "native"
    BORDERLESS_TABLE = "borderless_table"
    SCANNED = "scanned"


@dataclass
class ParsedDocument:
    path: Path
    blocks: list[Block]
    page_types: dict[int, PageType] = field(default_factory=dict)
    page_count: int = 0

    @property
    def parsers(self) -> Counter:
        return Counter(b.parser.value for b in self.blocks)


def classify_page(page: pdfplumber.page.Page) -> PageType:
    area = float(page.width * page.height) or 1.0
    # 只算落在页面里的部分：带出血、超出页面的图片不能把覆盖率算到 100% 以上
    w, h = float(page.width), float(page.height)
    image_area = sum(
        max(0.0, min(im["x1"], w) - max(im["x0"], 0.0)) * max(0.0, min(im["bottom"], h) - max(im["top"], 0.0))
        for im in page.images
    )
    if len(page.chars) < MIN_CHARS_PER_PAGE or image_area / area > IMAGE_COVERAGE_FOR_SCAN:
        return PageType.SCANNED
    return PageType.NATIVE


def _scan_page(page: pdfplumber.page.Page, page_no: int) -> list[Block]:
    scale = SCAN_RENDER_DPI / 72
    image = page.to_image(resolution=SCAN_RENDER_DPI).original.convert("RGB")
    return scanned.ocr_image(np.array(image), page_no, scale=scale)


def parse_pdf(path: Path, use_layout: bool = True) -> ParsedDocument:
    blocks: list[Block] = []
    page_types: dict[int, PageType] = {}
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_no = i + 1
            kind = classify_page(page)
            page_types[page_no] = kind
            if kind is PageType.SCANNED:
                blocks.extend(_scan_page(page, page_no))
                continue
            # 无边框表格以版面模型的 table 区域为准，所以只在开启版面分析时出现
            content = native_pdf.parse_page(page, page_no, use_layout=use_layout)
            if content.borderless:
                page_types[page_no] = PageType.BORDERLESS_TABLE
                table.rebuild_borderless(path, page, page_no, content.borderless)
            blocks.extend(content.blocks)
        page_count = len(pdf.pages)
    log.info("parsed %s: pages=%d types=%s", path.name, page_count, Counter(page_types.values()))
    return ParsedDocument(path=path, blocks=blocks, page_types=page_types, page_count=page_count)


# 手机拍的照片先按 EXIF 方向转正（纠偏只处理小角度倾斜，转不回 90°）；多页 TIFF 逐帧识别。
# 照片没有固定页边，不按上下边带判页眉页脚，免得把第一行、最后一行条款丢掉
def parse_image(path: Path) -> ParsedDocument:
    blocks: list[Block] = []
    page_types: dict[int, PageType] = {}
    with Image.open(path) as img:
        for page_no, frame in enumerate(ImageSequence.Iterator(img), start=1):
            rgb = np.array(ImageOps.exif_transpose(frame).convert("RGB"))
            blocks.extend(scanned.ocr_image(rgb, page_no, bands=False))
            page_types[page_no] = PageType.SCANNED
    return ParsedDocument(path=path, blocks=blocks, page_types=page_types, page_count=len(page_types))


def parse_document(path: Path, use_layout: bool = True) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, use_layout=use_layout)
    if suffix in IMAGE_SUFFIXES:
        return parse_image(path)
    raise ValueError(f"不支持的文件类型: {path.suffix}")


def body_blocks(doc: ParsedDocument) -> list[Block]:
    return [b for b in doc.blocks if b.kind not in (BlockKind.HEADER, BlockKind.FOOTER)]

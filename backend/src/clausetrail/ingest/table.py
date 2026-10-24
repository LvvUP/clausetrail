import html
import logging
import re
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pdfplumber

from clausetrail.ingest.native_pdf import RENDER_ZOOM, drop_superscripts
from clausetrail.types import Block, ParserName

log = logging.getLogger(__name__)
_warned_mineru = False

CROP_PADDING = 4.0
ROW_MATCH_THRESHOLD = 0.3
MIN_TSR_ROWS = 2
MIN_TSR_COLUMNS = 2

_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


def mineru_available() -> bool:
    try:
        import mineru  # noqa: F401
    except ImportError:
        return False
    return True


def _mineru_tables(pdf_path: Path, page_index: int) -> list[str]:
    from mineru.cli.common import do_parse, read_fn

    with tempfile.TemporaryDirectory() as out_dir:
        do_parse(
            output_dir=out_dir,
            pdf_file_names=[pdf_path.stem],
            pdf_bytes_list=[read_fn(pdf_path)],
            p_lang_list=["ch"],
            backend="pipeline",
            start_page_id=page_index,
            end_page_id=page_index,
            f_dump_md=True,
        )
        md_files = list(Path(out_dir).rglob("*.md"))
        if not md_files:
            return []
        markdown = md_files[0].read_text(encoding="utf-8")
    return [seg for seg in markdown.split("\n\n") if "<table" in seg]


def html_table_rows(segment: str) -> list[str]:
    rows = []
    for tr in _TR.findall(segment):
        cells = [html.unescape(_TAG.sub("", c)).strip() for c in _CELL.findall(tr)]
        if any(cells):
            rows.append("; ".join(c for c in cells if c))
    return rows


@lru_cache(maxsize=1)
def _tsr_model():
    from deepdoc.vision import TableStructureRecognizer

    return TableStructureRecognizer()


def _tag(box: dict, key: str, components: list[dict], index: int | None) -> None:
    if index is None:
        return
    c = components[index]
    if key == "R":
        box.update(R=index, R_top=c["top"], R_bott=c["bottom"])
    elif key == "C":
        box.update(C=index, C_left=c["x0"], C_right=c["x1"])
    else:
        box.update({key: index if key == "SP" else True})
        box.update(H_top=c["top"], H_bott=c["bottom"], H_left=c["x0"], H_right=c["x1"])


# 用 DeepDoc TSR 识别行/列/表头/合并单元格，再把 pdfplumber 的词框归到单元格里
def _tsr_rows(page: pdfplumber.page.Page, page_no: int, bbox: tuple[float, float, float, float]) -> list[str]:
    from deepdoc.vision import Recognizer, TableStructureRecognizer

    x0, top = max(bbox[0] - CROP_PADDING, 0.0), max(bbox[1] - CROP_PADDING, 0.0)
    crop = page.crop(
        (x0, top, min(bbox[2] + CROP_PADDING, float(page.width)), min(bbox[3] + CROP_PADDING, float(page.height)))
    )
    image = crop.to_image(resolution=72 * RENDER_ZOOM).annotated.convert("RGB")
    detected = _tsr_model()([np.array(image)])
    if not detected:
        return []
    components = detected[0]
    for c in components:
        c["x0"], c["x1"] = c["x0"] / RENDER_ZOOM + x0, c["x1"] / RENDER_ZOOM + x0
        c["top"], c["bottom"] = c["top"] / RENDER_ZOOM + top, c["bottom"] / RENDER_ZOOM + top
    rows = Recognizer.sort_Y_firstly([c for c in components if c["label"] == "table row"], 0)
    columns = Recognizer.sort_X_firstly([c for c in components if c["label"] == "table column"], 0)
    headers = [c for c in components if c["label"] == "table column header"]
    spans = [c for c in components if c["label"] == "table spanning cell"]
    if len(rows) < MIN_TSR_ROWS or len(columns) < MIN_TSR_COLUMNS:
        return []

    boxes = []
    for w in crop.extract_words():
        box = {k: w[k] for k in ("text", "x0", "x1", "top", "bottom")}
        box.update(page_number=page_no, layout_type="table")
        _tag(box, "R", rows, Recognizer.find_overlapped_with_threashold(box, rows, thr=ROW_MATCH_THRESHOLD))
        _tag(box, "H", headers, Recognizer.find_overlapped_with_threashold(box, headers, thr=ROW_MATCH_THRESHOLD))
        _tag(box, "C", columns, Recognizer.find_horizontally_tightest_fit(box, columns))
        _tag(box, "SP", spans, Recognizer.find_overlapped_with_threashold(box, spans, thr=ROW_MATCH_THRESHOLD))
        boxes.append(box)
    if not boxes:
        return []
    try:
        return [r for r in TableStructureRecognizer.construct_table(boxes, html=False) if r.strip()]
    except (AssertionError, IndexError, ValueError) as exc:
        log.debug("第 %d 页表格结构还原失败：%s", page_no, exc)
        return []


def _warn_mineru_once() -> None:
    global _warned_mineru
    if not _warned_mineru:
        _warned_mineru = True
        log.warning("mineru 未安装，无边框表格改用 DeepDoc TSR 还原（pip install '.[mineru]' 可启用 MinerU）")


def rebuild_borderless(pdf_path: Path, page: pdfplumber.page.Page, page_no: int, blocks: list[Block]) -> None:
    """就地改写 blocks 的文本；两种还原都失败时保留按行拼接的原文。"""
    if not blocks:
        return
    if mineru_available():
        try:
            segments = _mineru_tables(pdf_path, page_no - 1)
        except Exception:
            # MinerU 出错只影响这一页的表格还原：改用 DeepDoc TSR，不中止整份文档入库
            log.warning("第 %d 页 MinerU 还原表格失败，改用 DeepDoc TSR", page_no, exc_info=True)
            segments = None
        if segments is not None and len(segments) == len(blocks):
            for block, segment in zip(blocks, segments, strict=True):
                rows = html_table_rows(segment)
                if rows:
                    block.text, block.parser = "\n".join(rows), ParserName.MINERU
            return
        if segments is not None:
            log.warning(
                "第 %d 页 MinerU 识别出 %d 个表格、版面模型为 %d 个，改用 DeepDoc TSR",
                page_no,
                len(segments),
                len(blocks),
            )
    else:
        _warn_mineru_once()
    page = drop_superscripts(page)
    for block in blocks:
        try:
            rows = _tsr_rows(page, page_no, block.bbox)
        except Exception:
            # 一张表还原失败只保留它按行拼接的原文，不影响同页的其他表格
            log.warning("第 %d 页表格结构还原出错，保留原文", page_no, exc_info=True)
            continue
        if rows:
            block.text, block.parser = "\n".join(rows), ParserName.DEEPDOC_TSR_FALLBACK

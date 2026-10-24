import bisect
import re
import statistics
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pdfplumber

from clausetrail.structure.heading_features import numbering
from clausetrail.types import Block, BlockKind, ParserName

RENDER_ZOOM = 3
LAYOUT_THRESHOLD = 0.2
TRUSTED_REGION_SCORE = 0.5
MIN_TABLE_FILL_RATIO = 0.6
SYMBOL_FONTS = ("Wingdings", "Symbol", "ZapfDingbats")
HEADER_FOOTER_BAND = 0.06
SUPERSCRIPT_RATIO = 0.6
# 字间距超过 0.9 个字号视为栏间距；超过 3pt（pdfplumber 默认 x_tolerance）补一个空格
GUTTER_RATIO = 0.9
WORD_GAP = 3.0
SIDEBAR_WRAP_MAX_CHARS = 8
# 出现这些编号的「表格」区域其实是「条号 | 正文」两栏排版
CLAUSE_STYLES = {"chapter_cn", "article_cn", "decimal_2", "decimal_3"}
PRIVATE_USE_START, PRIVATE_USE_END = 0xE000, 0xF8FF
CJK = re.compile(r"[一-鿿]")
NUMBER_TOKEN = re.compile(
    r"^(\d{1,2}(\.\d{1,2}){0,3}\.?|[（(][一二三四五六七八九十\d]{1,3}[）)]|[一二三四五六七八九十]+、)$"
)

_LAYOUT_TO_KIND = {
    "title": BlockKind.TITLE,
    "text": BlockKind.TEXT,
    "table": BlockKind.TABLE,
    "figure": BlockKind.FIGURE,
    "header": BlockKind.HEADER,
    "footer": BlockKind.FOOTER,
}


@dataclass
class Region:
    kind: BlockKind
    bbox: tuple[float, float, float, float]
    score: float


@lru_cache(maxsize=1)
def _layout_model():
    from deepdoc.vision import LayoutRecognizer

    return LayoutRecognizer("layout")


def detect_regions(page: pdfplumber.page.Page) -> list[Region]:
    from deepdoc.vision import Recognizer

    image = page.to_image(resolution=72 * RENDER_ZOOM).annotated.convert("RGB")
    raw = Recognizer.__call__(_layout_model(), [np.array(image)], thr=LAYOUT_THRESHOLD)[0]
    regions = []
    for r in raw:
        kind = _LAYOUT_TO_KIND.get(r["type"].lower())
        if kind is None or float(r["score"]) < TRUSTED_REGION_SCORE:
            continue
        x0, y0, x1, y1 = (v / RENDER_ZOOM for v in r["bbox"])
        regions.append(Region(kind, (x0, y0, x1, y1), float(r["score"])))
    return _dedupe(regions)


def _dedupe(regions: list[Region]) -> list[Region]:
    kept: list[Region] = []
    for r in sorted(regions, key=lambda r: -r.score):
        if any(_iou(r.bbox, k.bbox) > 0.7 for k in kept):
            continue
        kept.append(r)
    return kept


def _iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _region_for(line_bbox, regions: list[Region]) -> Region | None:
    x0, top, x1, bottom = line_bbox
    cx, cy = (x0 + x1) / 2, (top + bottom) / 2
    hits = [r for r in regions if r.bbox[0] - 2 <= cx <= r.bbox[2] + 2 and r.bbox[1] - 2 <= cy <= r.bbox[3] + 2]
    if not hits:
        return None
    return min(hits, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))


BOLD_WEIGHT = re.compile(r"bold|heavy|black|[-,]b$")
LIGHT_WEIGHT = re.compile(r"light|thin")


# 按字重词判断，不按子串：「-b」会命中 Helvetica-Book。黑体类（SimHei、STHeiti）在条款里多用来排标题，算粗体；
# 微软雅黑（YaHei）是常规字重，不算；黑体的细体（STHeiti Light、JhengHei Light）也不算
def _is_bold(fontname: str) -> bool:
    name = fontname.lower()
    if BOLD_WEIGHT.search(name):
        return True
    return "hei" in name and "yahei" not in name and not LIGHT_WEIGHT.search(name)


def _keep_char(c: dict) -> bool:
    if any(s in c["fontname"] for s in SYMBOL_FONTS):
        return False
    return not (c["text"] and PRIVATE_USE_START <= ord(c["text"][0]) <= PRIVATE_USE_END)


def _visible(chars: list[dict]) -> list[dict]:
    return [c for c in chars if c["text"].strip() and not c.get("ghost")]


def _segments(chars: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    last = None
    for c in chars:
        if not c["text"].strip():
            if out:
                out[-1].append(c)
            continue
        if last is None or c["x0"] - last["x1"] > max(last["size"], c["size"]) * GUTTER_RATIO:
            out.append([c])
        else:
            out[-1].append(c)
        last = c
    return [s for s in out if _visible(s)]


def _join(chars: list[dict]) -> str:
    out: list[str] = []
    prev, pending_space = None, False
    for c in chars:
        if not c["text"].strip():
            pending_space = prev is not None
            continue
        if c.get("ghost"):
            # 角标不输出，但占住位置，免得它留下的空隙被当成词间空格
            if prev is not None:
                prev = c
            continue
        if prev is not None and (pending_space or c["x0"] - prev["x1"] > WORD_GAP):
            out.append(" ")
        out.append(c["text"])
        prev, pending_space = c, False
    return "".join(out)


def _line(chars: list[dict]) -> dict:
    visible = _visible(chars)
    return {
        "text": _join(chars),
        "bbox": (
            min(c["x0"] for c in visible),
            min(c["top"] for c in visible),
            max(c["x1"] for c in visible),
            max(c["bottom"] for c in visible),
        ),
        "font_size": statistics.median(c["size"] for c in visible),
        "bold": sum(_is_bold(c["fontname"]) for c in visible) / len(visible) > 0.5,
    }


def _continues_head(head: dict, segment: list[dict], text: str) -> bool:
    x0, _, x1, bottom = head["bbox"]
    visible = _visible(segment)
    left, top = min(c["x0"] for c in visible), min(c["top"] for c in visible)
    return (
        len(text) <= SIDEBAR_WRAP_MAX_CHARS
        and bool(CJK.search(text))
        and numbering(text)[0] is None
        and x0 - 2 <= left <= x1
        and 0 <= top - bottom <= head["font_size"]
    )


# 有的条款是「编号 | 标题 | 正文」三栏排版，pdfplumber 会把同一行的三栏拼成一行；
# 这里按栏间距拆开，把「编号+标题」单独成行，标题折到下一行的几个字再接回去
def page_lines(page: pdfplumber.page.Page) -> list[dict]:
    floor = superscript_floor(page)
    lines: list[dict] = []
    head: list[dict] | None = None
    head_at = -1
    for ln in page.extract_text_lines(return_chars=True, strip=True):
        kept = [c for c in ln["chars"] if _keep_char(c)]
        # 整行都是小字（小字号表格、脚注）时没有角标可言，行里的数字和 % 要保留
        has_body = any(c["size"] >= floor for c in kept if c["text"].strip())
        chars = [{**c, "ghost": True} if has_body and is_superscript(c, floor) else c for c in kept]
        segments = _segments(chars)
        if not segments:
            continue
        texts = [_join(s) for s in segments]
        multi = len(segments) > 1
        if head is not None and _continues_head(_line(head), segments[0], texts[0]):
            head = head + segments[0]
            lines[head_at] = _line(head)
            rest = segments[1:]
        elif multi and NUMBER_TOKEN.match(texts[0]) and CJK.search(texts[1]):
            head = segments[0] + segments[1]
            lines.append(_line(head))
            head_at = len(lines) - 1
            rest = segments[2:]
        else:
            # 正文栏里的行不打断左栏标题的折行，回到左栏才算标题结束
            if head is not None and _visible(segments[0])[0]["x0"] <= _line(head)["bbox"][2]:
                head = None
            if multi and all(numbering(t)[0] for t in texts):
                lines.extend(_line(s) for s in segments)
                continue
            rest = segments
        if rest:
            lines.append(_line([c for s in rest for c in s]))
    return lines


def _table_blocks(page: pdfplumber.page.Page, page_no: int) -> list[Block]:
    blocks = []
    for table in page.find_tables():
        rows = table.extract()
        if not rows or len(rows) < 2:
            continue
        cells = [c for row in rows for c in row]
        if sum(1 for c in cells if c and c.strip()) / len(cells) < MIN_TABLE_FILL_RATIO:
            continue
        lines = [" | ".join((cell or "").replace("\n", "") for cell in row) for row in rows]
        blocks.append(
            Block(
                page=page_no,
                bbox=tuple(table.bbox),
                text="\n".join(lines),
                kind=BlockKind.TABLE,
                parser=ParserName.DEEPDOC,
            )
        )
    return blocks


def _inside(bbox, container) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return container[0] <= cx <= container[2] and container[1] <= cy <= container[3]


def _covered(inner, outer) -> float:
    ix0, iy0, ix1, iy1 = (
        max(inner[0], outer[0]),
        max(inner[1], outer[1]),
        min(inner[2], outer[2]),
        min(inner[3], outer[3]),
    )
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0) / area if area > 0 else 0.0


@dataclass
class PageContent:
    blocks: list[Block]
    # 版面模型判为表格、但 pdfplumber 找不到框线的区域；由 table.rebuild_borderless 还原单元格
    borderless: list[Block] = field(default_factory=list)


def superscript_floor(page: pdfplumber.page.Page) -> float:
    sizes = [c["size"] for c in page.chars if c["text"].strip()]
    return statistics.median(sizes) * SUPERSCRIPT_RATIO if sizes else 0.0


# 条款里的脚注角标（如「永久不可逆53性」）字号远小于正文，混进正文会污染检索和标题特征
def is_superscript(c: dict, floor: float) -> bool:
    return c["size"] < floor and not c["text"].isalpha()


# 正文字占据的纵向区间（相互重叠的合并），按 top 排好序
def _body_bands(chars: list[dict], floor: float) -> list[tuple[float, float]]:
    bands: list[list[float]] = []
    for top, bottom in sorted((c["top"], c["bottom"]) for c in chars if c["size"] >= floor and c["text"].strip()):
        if bands and top <= bands[-1][1]:
            bands[-1][1] = max(bands[-1][1], bottom)
        else:
            bands.append([top, bottom])
    return [(a, b) for a, b in bands]


def _overlaps_body(c: dict, bands: list[tuple[float, float]], tops: list[float]) -> bool:
    i = bisect.bisect_right(tops, c["bottom"]) - 1
    return i >= 0 and bands[i][1] >= c["top"]


# 公式里的上下标（T₁N₀M₀、10⁹/L、O₂、「（1＋年利率）」的 1/360 次方）紧跟在字母、数字或右括号后面，是要保留的内容；
# 脚注角标跟在汉字或中文标点后面
FORMULA_ANCHOR = re.compile(r"[A-Za-z0-9)）\]]")


# 同一行里紧挨在 c 左边的字
def _left_neighbor(c: dict, chars: list[dict], gap: float) -> dict | None:
    best = None
    for o in chars:
        if o is c or not o["text"].strip():
            continue
        beside = o["x1"] <= c["x0"] + 1 and c["x0"] - o["x1"] < gap
        if beside and o["top"] <= c["bottom"] and o["bottom"] >= c["top"] and (best is None or o["x1"] > best["x1"]):
            best = o
    return best


def _footnote_marks(chars: list[dict], floor: float) -> set[int]:
    """和正文同一行的小字里真正的脚注角标（返回 id）。连着的一串小字（如 1/360）按这串前面的正文字判断。"""
    bands = _body_bands(chars, floor)
    tops = [b[0] for b in bands]
    small = [c for c in chars if is_superscript(c, floor) and _overlaps_body(c, bands, tops)]
    small_ids = {id(c) for c in small}
    anchors: dict[int, str] = {}

    def anchor(c: dict) -> str:
        key = id(c)
        if key not in anchors:
            anchors[key] = ""  # 先占位，防止互相引用时无限递归
            n = _left_neighbor(c, chars, floor)
            anchors[key] = "" if n is None else anchor(n) if id(n) in small_ids else n["text"]
        return anchors[key]

    return {id(c) for c in small if not FORMULA_ANCHOR.fullmatch(anchor(c))}


# 只删和正文字在同一行的小字：小字号表格、脚注里整行都是小字，里面的数字和 % 是要保留的内容；
# 同一行里跟在字母、数字后面的上下标也保留
def drop_superscripts(page: pdfplumber.page.Page) -> pdfplumber.page.Page:
    floor = superscript_floor(page)
    if not floor:
        return page
    marks = _footnote_marks(page.chars, floor)
    if not marks:
        return page
    return page.filter(lambda obj: obj.get("object_type") != "char" or id(obj) not in marks)


def _group_table(lines: list[Block], page_no: int) -> Block:
    return Block(
        page=page_no,
        bbox=(
            min(b.bbox[0] for b in lines),
            min(b.bbox[1] for b in lines),
            max(b.bbox[2] for b in lines),
            max(b.bbox[3] for b in lines),
        ),
        text="\n".join(b.text for b in lines),
        kind=BlockKind.TABLE,
        font_size=statistics.median(b.font_size for b in lines),
        parser=ParserName.DEEPDOC,
    )


def parse_page(
    page: pdfplumber.page.Page, page_no: int, use_layout: bool = True, regions: list[Region] | None = None
) -> PageContent:
    if regions is None:
        regions = detect_regions(page) if use_layout else []
    lines = page_lines(page)
    page = drop_superscripts(page)
    tables = _table_blocks(page, page_no)
    ruled = {
        id(r) for r in regions if r.kind == BlockKind.TABLE and any(_covered(t.bbox, r.bbox) > 0.5 for t in tables)
    }
    height = float(page.height)
    left_margin = min((ln["bbox"][0] for ln in lines), default=0.0)
    placed = [
        (ln, _region_for(ln["bbox"], regions)) for ln in lines if not any(_inside(ln["bbox"], t.bbox) for t in tables)
    ]
    # 版面模型偶尔把「条号 | 正文」两栏排版整块判成表格；含条款编号的区域按正文处理
    as_text = ruled | {
        id(r)
        for ln, r in placed
        if r is not None and r.kind == BlockKind.TABLE and numbering(ln["text"])[0] in CLAUSE_STYLES
    }

    blocks: list[Block] = []
    table_lines: dict[int, list[Block]] = {}
    for ln, region in placed:
        kind = region.kind if region else BlockKind.TEXT
        if kind == BlockKind.TABLE and id(region) in as_text:
            kind = BlockKind.TEXT
        top, bottom = ln["bbox"][1], ln["bbox"][3]
        in_band = bottom < height * HEADER_FOOTER_BAND or top > height * (1 - HEADER_FOOTER_BAND)
        if kind in (BlockKind.TEXT, BlockKind.TABLE) and in_band:
            kind = BlockKind.HEADER if top < height / 2 else BlockKind.FOOTER
        block = Block(
            page=page_no,
            bbox=ln["bbox"],
            text=ln["text"],
            kind=kind,
            font_size=ln["font_size"],
            bold=ln["bold"],
            indent=ln["bbox"][0] - left_margin,
            parser=ParserName.DEEPDOC,
        )
        if kind == BlockKind.TABLE:
            table_lines.setdefault(id(region), []).append(block)
        else:
            blocks.append(block)

    borderless = [_group_table(group, page_no) for group in table_lines.values()]
    blocks.extend(borderless)
    blocks.extend(tables)
    blocks.sort(key=lambda b: (round(b.bbox[1] / 4), b.bbox[0]))
    return PageContent(blocks, borderless)

import statistics
from pathlib import Path

import pdfplumber

from clausetrail.chunking.merge_rules import Unit
from clausetrail.chunking.splitter import _split_long
from clausetrail.chunking.text_utils import count_tokens
from clausetrail.ingest.native_pdf import SUPERSCRIPT_RATIO, Region, drop_superscripts, page_lines, parse_page
from clausetrail.ingest.table import html_table_rows
from clausetrail.structure.heading_features import rule_labels
from clausetrail.types import Block, BlockKind

SAMPLES = Path(__file__).resolve().parents[3] / "data" / "samples"


def _line(text: str, x0: float, x1: float, top: float) -> Block:
    return Block(page=1, bbox=(x0, top, x1, top + 12), text=text, font_size=10.5)


def test_wrapped_numbered_lines_and_lead_ins_are_body():
    blocks = [
        _line("1. 保险费率调整的触发条件", 60, 200, 100),
        _line("3. 疗养、视力矫正手术、各种健康体检项目及预防性医疗项目、牙科保健、牙", 60, 540, 130),
        _line("齿美容、视力矫正等。", 60, 200, 145),
        _line("49 现金价值的计算分两种情况：", 60, 210, 170),
        _line("本合同的现金价值按下列方式计算，具体数额以保险单载明的为准，请您仔细阅读。", 60, 540, 190),
    ]
    assert rule_labels(blocks) == [1, 0, 0, 0, 0]


def test_long_table_splits_on_row_boundaries():
    rows = [f"{i}; 疾病名称{i}; " + "指经专科医生确诊的疾病" * 6 for i in range(1, 30)]
    pieces = _split_long(Unit("\n".join(rows), 16, 16, is_table=True), 200)
    assert len(pieces) > 1
    assert all(p.is_table and count_tokens(p.text) <= 200 for p in pieces)
    assert [r for p in pieces for r in p.text.split("\n")] == rows


def test_mineru_html_rows():
    segment = (
        "<table><tr><th>疾病</th><th>定义</th></tr>"
        "<tr><td>恶性肿瘤&mdash;重度</td><td><b>指</b>恶性细胞不受控制地进行性增长</td></tr>"
        "<tr><td></td><td></td></tr></table>"
    )
    assert html_table_rows(segment) == ["疾病; 定义", "恶性肿瘤—重度; 指恶性细胞不受控制地进行性增长"]


def test_layout_table_region_becomes_one_block():
    with pdfplumber.open(SAMPLES / "chinalife-whole-life-hengyu-yisheng.pdf") as pdf:
        page = pdf.pages[0]
        plain = parse_page(page, 1, use_layout=False).blocks
        target = [b for b in plain if b.kind == BlockKind.TEXT][2:6]
        box = (
            min(b.bbox[0] for b in target),
            target[0].bbox[1],
            max(b.bbox[2] for b in target),
            target[-1].bbox[3],
        )
        content = parse_page(page, 1, regions=[Region(BlockKind.TABLE, box, 0.9)])
    assert len(content.borderless) == 1
    assert [b for b in content.blocks if b.kind == BlockKind.TABLE] == content.borderless
    assert content.borderless[0].text == "\n".join(b.text for b in target)
    assert len(content.blocks) == len(plain) - len(target) + 1


def test_sidebar_headings_are_split_from_body():
    with pdfplumber.open(SAMPLES / "pingan-long-term-medical-anyibao-exiang.pdf") as pdf:
        texts = [ln["text"] for ln in page_lines(pdf.pages[8])]
    assert "1.9 保险金给付限额" in texts
    assert "1.10 保险期间和续保" in texts
    assert "我们不保什么" in texts
    assert not any("" in t for t in texts)
    # 角标「21」被去掉后不能在正文里留下空格
    assert any("遗传性疾病，先天性畸形" in t for t in texts)


def test_table_region_with_clause_numbers_is_body():
    with pdfplumber.open(SAMPLES / "pingan-long-term-medical-anyibao-exiang.pdf") as pdf:
        content = parse_page(pdf.pages[1], 2, regions=[Region(BlockKind.TABLE, (49, 173, 548, 737), 0.9)])
    assert not content.borderless
    assert any(b.kind == BlockKind.TEXT and b.text.startswith("1.1 保险计划") for b in content.blocks)


def test_footnote_superscripts_are_dropped():
    with pdfplumber.open(SAMPLES / "pingan-long-term-medical-anyibao-exiang.pdf") as pdf:
        page = pdf.pages[16]
        floor = statistics.median(c["size"] for c in page.chars if c["text"].strip()) * SUPERSCRIPT_RATIO
        filtered = drop_superscripts(page)

        def tiny_digits(p) -> int:
            return sum(1 for c in p.chars if c["size"] < floor and c["text"].isdigit())

        def cjk(p) -> int:
            return sum(1 for c in p.chars if "一" <= c["text"] <= "鿿")

        assert tiny_digits(page) > 0
        assert tiny_digits(filtered) == 0
        assert cjk(filtered) == cjk(page)

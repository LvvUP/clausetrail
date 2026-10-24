import pytest

from clausetrail import lexicon
from clausetrail.chunking.merge_rules import Unit, merge_structural
from clausetrail.chunking.splitter import _split_long, chunk_document, is_toc_like
from clausetrail.chunking.text_utils import count_tokens
from clausetrail.structure.heading_features import numbering
from clausetrail.structure.ocr_correction import correct_text, load_confusions, load_terms
from clausetrail.types import Block, BlockKind, SectionNode


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("第一章 总则", ("chapter_cn", 1)),
        ("第五条 等待期", ("article_cn", 2)),
        ("1.6.1 一般医疗保险金", ("decimal_3", 3)),
        ("（一）投保人对被保险人的故意杀害", ("paren_cn", 3)),
        ("90日内", (None, 0)),
        ("2026年起", (None, 0)),
    ],
)
def test_numbering(text, expected):
    assert numbering(text) == expected


def test_list_lead_in_keeps_items_together():
    units = [
        Unit(t, 1, 1)
        for t in ["因下列情形之一导致身故的，我们不承担责任：", "1．故意杀害；", "2．故意犯罪；", "其他内容。"]
    ]
    merged = merge_structural(units, max_tokens=480)
    assert len(merged) == 2
    assert "1．故意杀害" in merged[0].text and "2．故意犯罪" in merged[0].text


def test_proviso_joins_previous_sentence():
    units = [
        Unit("被保险人自本合同成立之日起2年内自杀的，我们不承担责任。", 1, 1),
        Unit("但被保险人自杀时为无民事行为能力人的除外。", 1, 1),
    ]
    merged = merge_structural(units, max_tokens=480)
    assert len(merged) == 1 and merged[0].text.endswith("除外。")


def test_toc_detection():
    assert is_toc_like(
        "第一条 合同构成 第二十二条 联系方式变更第二条 投保范围 第二十三条 年龄错误的处理第三条 合同成立"
    )
    assert not is_toc_like("第七条 责任免除\n因下列情形之一导致被保险人身故的，我们不承担给付身故保险金的责任。")


def _block(text: str, page: int = 1) -> Block:
    return Block(page=page, bbox=(0, 0, 100, 10), text=text)


def test_chunks_respect_limit_and_overlap():
    sentence = "被保险人在等待期内因疾病确诊重大疾病的，我们无息退还已交保险费，本合同终止。"
    node = SectionNode(title="第五条 等待期", level=2, page=1, blocks=[_block(sentence * 6)])
    root = SectionNode(title="", level=0, page=1, children=[node])
    chunks = chunk_document(root, "d1", "x.pdf", "deepdoc", max_tokens=120, overlap_sentences=1)
    assert len(chunks) >= 2
    assert all(count_tokens(c.text) <= 120 for c in chunks)
    assert all(c.section_path == ["第五条 等待期"] for c in chunks)
    assert chunks[1].text.startswith(sentence)
    assert [c.char_start for c in chunks] == sorted(c.char_start for c in chunks)


def test_numeric_table_is_not_dropped_as_toc():
    table = "家庭成员人数 | 家庭费率因子\n1人 | 100%\n2人 | 95%\n3人 | 90%\n4人及以上 | 85%"
    assert is_toc_like(table)  # 格式上确实像目录，所以必须按表格身份放行
    block = Block(page=3, bbox=(0, 0, 100, 40), text=table, kind=BlockKind.TABLE)
    node = SectionNode(title="1.7 保险金计算方法", level=2, page=3, blocks=[block])
    chunks = chunk_document(SectionNode(title="", level=0, page=1, children=[node]), "d1", "x.pdf", "deepdoc")
    assert any("4人及以上 | 85%" in c.text for c in chunks)


def test_split_pieces_keep_their_own_pages():
    sentence = "被保险人在等待期内因疾病确诊重大疾病的，我们无息退还已交保险费，本合同终止。"
    blocks = [_block(sentence * 4, page=p) for p in (1, 2, 3)]
    unit = Unit("".join(b.text for b in blocks), 1, 3, blocks)
    pieces = _split_long(unit, 120)
    assert len(pieces) >= 3
    assert (pieces[0].page_start, pieces[0].page_end) == (1, 1)
    assert (pieces[-1].page_start, pieces[-1].page_end) == (3, 3)
    assert all(set(map(id, p.blocks)) <= set(map(id, blocks)) for p in pieces)


def test_ocr_correction_fixes_confusable_terms():
    fixed, fixes = correct_text("被保险人在等待斯内确诊重大痰病", load_terms(), load_confusions())
    assert fixed == "被保险人在等待期内确诊重大疾病"
    assert ("等待斯", "等待期") in fixes
    assert correct_text("本合同的等待期为90日", load_terms(), load_confusions())[1] == []


def test_query_expansion_adds_canonical_terms():
    tokens = lexicon.query_tokens("重疾险的观察期是多久").split()
    assert "重大疾病" in tokens and "等待期" in tokens
    assert "的" not in tokens

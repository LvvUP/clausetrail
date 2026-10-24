from clausetrail.types import Chunk, SectionNode
from eval.gen.candidates import _definition, detail_candidates
from eval.gen.check_leakage import check
from eval.gen.corpus import Doc, assign_splits
from eval.gen.llm_gen import parse_generated
from eval.metrics import best_threshold, is_hit, norm, ranking_metrics, reciprocal_rank, truncation_rate


def test_ranking_metrics():
    assert reciprocal_rank([False, True]) == 0.5
    assert reciprocal_rank([False] * 10 + [True]) == 0.0
    m = ranking_metrics([[True], [False, False, True], [False] * 6 + [True]])
    assert m["mrr@10"] == round((1 + 1 / 3 + 1 / 7) / 3, 4)
    assert m["recall@5"] == round(2 / 3, 4)
    assert m["recall@10"] == 1.0


def test_hit_ignores_whitespace_and_checks_document():
    gold = [{"doc_id": "d1", "span": "犹豫期为 15 日。"}]
    assert is_hit("d1", "……犹豫期为15日。\n退保……", gold)
    assert not is_hit("d2", "犹豫期为15日。", gold)


def test_truncation_rate():
    assert truncation_rate(["甲乙丙。", "丁戊己。"], ["甲乙丙。丁", "戊己。"]) == {
        "units": 2,
        "truncated": 1,
        "rate": 0.5,
    }


def test_best_threshold_separates_and_prefers_lower():
    fit = best_threshold([0.9, 0.8, 0.7], [0.2, 0.3])
    assert fit["balanced_accuracy"] == 1.0
    assert 0.3 < fit["threshold"] < 0.7
    assert best_threshold([0.5], []) is None


def test_leakage_check():
    dev = [
        {"qid": "dev-1", "query": "恒裕一生的犹豫期是怎么规定的？", "gold": [{"doc_id": "a", "span": "犹豫期为15日。"}]}
    ]
    test = [
        {"qid": "rt-1", "query": "恒裕一生的犹豫期是怎么规定的？", "gold": [{"doc_id": "a", "span": "其他内容。"}]},
        {"qid": "rt-2", "query": "康宁悦享的等待期是多久？", "gold": [{"doc_id": "b", "span": "等待期为90日。"}]},
    ]
    problems = check(dev, {"retrieval_test": test})
    assert any("rt-1" in p and "共享文档" in p for p in problems)
    assert any("rt-1" in p and "近似重复" in p for p in problems)
    assert not any("rt-2" in p for p in problems)


def test_generated_questions_keep_verbatim_spans_only():
    chunk = "本合同的犹豫期为15日。等待期为90日。"
    output = '说明\n[{"question": "犹豫期多久？", "span": "犹豫期为15日。"}, {"question": "x", "span": "编造的句子。"}]'
    assert parse_generated(output, chunk) == [{"question": "犹豫期多久？", "span": "犹豫期为15日。"}]
    assert parse_generated("不是 JSON", chunk) == []


def test_definition_requires_term_right_before_zhi():
    assert _definition("犹豫期指您签收合同后的15日内的期间。", "犹豫期").startswith("犹豫期指")
    row = "4; 冠状动脉搭桥术（或称冠状动脉旁路移植术）*; 指为治疗严重的冠心病实施的手术。"
    assert _definition(row, "冠状动脉搭桥术").startswith("冠状动脉搭桥术")
    assert _definition("护理费：指住院期间由专业护士提供临床护理服务所收取的费用。", "护理") is None
    assert _definition("艾滋病病毒指人类免疫缺陷病毒，英文缩写为HIV。", "艾滋病") is None
    assert _definition("身故保险金指被保险人身故时给付的保险金。", "保险金") is None


def _chunk(cid: str, title: str, text: str) -> Chunk:
    return Chunk(cid, "d", "f.pdf", text, ["测试条款", title], 1, 1, 0, len(text), "deepdoc")


def test_detail_candidates_one_question_per_gold_sentence():
    doc = Doc(
        "d",
        "f.pdf",
        "测试产品终身寿险",
        "测试产品",
        SectionNode("", 0, 1),
        [
            _chunk(
                "c1",
                "第七条 保险责任",
                "保险费应按年交纳，交费期间为十年。未按时交纳保险费的，合同效力中止。"
                "犹豫期为15日，期间解除合同可退还全部款项。",
            ),
            _chunk("c2", "第五条 疾病定义", "观察期内初次确诊的疾病不属于保障范围。"),
        ],
    )
    surfaces = {"保险费": "保险费", "犹豫期": "犹豫期", "等待期": "等待期", "观察期": "等待期"}
    used: set[str] = set()
    cands = detail_candidates(doc, surfaces, used)
    # 「保险费」在同一节出现在两句话里，指不到唯一一句，不出题；别名「观察期」按标准术语「等待期」出题
    assert {c.key for c in cands} == {"犹豫期", "等待期"}
    assert len({c.span for c in cands}) == len(cands)
    assert {norm(c.span) for c in cands} <= used


def test_leakage_flags_two_questions_on_one_gold_sentence():
    rows = [
        {"qid": "rt-1", "query": "犹豫期多久？", "gold": [{"doc_id": "a", "span": "犹豫期为15日。"}]},
        {"qid": "rt-2", "query": "退保期限是几天？", "gold": [{"doc_id": "a", "span": "犹豫期为 15 日。"}]},
    ]
    assert any("同一句 gold" in p for p in check([], {"retrieval_test": rows}))


def test_splits_are_deterministic():
    ids = ["a", "b", "c", "d"]
    splits = assign_splits(ids)
    assert splits == assign_splits(list(reversed(ids)))
    assert list(splits.values()).count("dev") == 1

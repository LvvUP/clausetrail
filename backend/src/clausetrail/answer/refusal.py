from dataclasses import dataclass

from clausetrail.retrieval.pipeline import RetrievalResult

NO_HIT = "no_hit"
LOW_SCORE = "low_score"
UNSUPPORTED = "unsupported"

MESSAGES = {
    NO_HIT: "条款库里没有检索到与这个问题相关的内容。",
    LOW_SCORE: "检索到的条款与问题的相关度不够，无法据此作答。",
    UNSUPPORTED: "回答里没有能对应到条款原文的引用，已撤回这段回答。",
}


@dataclass
class Refusal:
    reason: str
    message: str
    top_score: float


# 拒答分两道：生成前看有没有召回、最高精排分够不够（precheck）；
# 生成后看回答里有没有有效引用编号（unsupported，由 chat 调用）。后一种发生时回答已经流式发出，
# refusal 事件表示撤回：前端撤下已显示的文字，库里也不保存
def precheck(result: RetrievalResult, threshold: float) -> Refusal | None:
    if not result.evidence:
        return Refusal(NO_HIT, MESSAGES[NO_HIT], 0.0)
    if result.top_score < threshold:
        return Refusal(LOW_SCORE, MESSAGES[LOW_SCORE], result.top_score)
    return None


def unsupported(top_score: float) -> Refusal:
    return Refusal(UNSUPPORTED, MESSAGES[UNSUPPORTED], top_score)

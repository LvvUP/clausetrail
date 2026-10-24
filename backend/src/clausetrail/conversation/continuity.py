import re
from dataclasses import dataclass

from clausetrail.models.embedder import Embedder

PRONOUN = re.compile(
    r"(它们|它|这个|那个|这款|那款|这种|那种|该产品|该险种|该条款|该保险|上面|上述|刚才|前面|这一条|那一条|这项|那项|此款|这份|那份)"
)
ELLIPSIS = re.compile(r"^(那|那么|还有|另外|如果是|要是|换成|改成|再问)|呢[？?]?$")
FRAGMENT_CHARS = 5
SHORT_QUESTION_CHARS = 10

NEW, CONTINUE, SWITCH = "new", "continue", "switch"


@dataclass
class Decision:
    topic: str
    rewrite: bool
    reason: str
    similarity: float | None = None


async def judge(question: str, last_standalone: str | None, embedder: Embedder, threshold: float) -> Decision:
    if not last_standalone:
        return Decision(NEW, False, "first_turn")
    q = question.strip()
    # 指代、省略、过短这几条规则先判，命中就一定要改写，不用再算向量
    if PRONOUN.search(q):
        return Decision(CONTINUE, True, "pronoun")
    if ELLIPSIS.search(q):
        return Decision(CONTINUE, True, "ellipsis")
    if len(q) <= FRAGMENT_CHARS:
        return Decision(CONTINUE, True, "fragment")
    vecs = await embedder.embed([q, last_standalone])
    # embedder 输出的是单位向量，点积就是余弦相似度
    sim = float(vecs[0] @ vecs[1])
    if sim >= threshold:
        # 话题相近但问句已经够长的，多半本身就完整，只对短问句做改写
        short = len(q) <= SHORT_QUESTION_CHARS
        return Decision(CONTINUE, short, "similar_short" if short else "similar", round(sim, 4))
    return Decision(SWITCH, False, "dissimilar", round(sim, 4))

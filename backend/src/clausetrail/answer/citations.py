import re

MARKER = re.compile(r"\[(\d{1,2})\]")
# 模型写出的引用不一定是标准的 [1]：有全角的［1］【1】，也有合写的 [1,3]、[1、3]。
# 只把这几种改写成 [1][3]，正文里别的括号（如「【重要提示】」）原样保留
LOOSE_MARKER = re.compile(r"[\[［【](\d{1,2}(?:\s*[,，、]\s*\d{1,2})*)[\]］】]")
LIST_SEP = re.compile(r"\s*[,，、]\s*")
# 末尾还没闭合的半截标记；长度上限放宽到能装下 [1,2,3,4,5,6,7,8] 这类合写，多扣住几个字的代价很小
PARTIAL_TAIL = re.compile(r"[\[［【][\d,，、\s]{0,24}$")


def normalize_markers(text: str) -> str:
    return LOOSE_MARKER.sub(lambda m: "".join(f"[{n}]" for n in LIST_SEP.split(m.group(1))), text)


def used_citations(text: str, evidence_count: int) -> list[int]:
    seen: list[int] = []
    for m in MARKER.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= evidence_count and n not in seen:
            seen.append(n)
    return seen


# 编号超出证据条数的引用对应不到任何条款原文，直接删掉
def strip_invalid(text: str, evidence_count: int) -> str:
    return MARKER.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= evidence_count else "", text)


class CitationFilter:
    def __init__(self, evidence_count: int):
        self._n = evidence_count
        self._pending = ""
        self.text = ""

    def feed(self, delta: str) -> str:
        buf = self._pending + delta
        # 流式输出时 [12]、[1,3] 可能被拆进两个 delta，末尾的半截标记先留着，等下一段拼上再判断
        tail = PARTIAL_TAIL.search(buf)
        if tail:
            self._pending = buf[tail.start() :]
            buf = buf[: tail.start()]
        else:
            self._pending = ""
        out = strip_invalid(normalize_markers(buf), self._n)
        self.text += out
        return out

    def flush(self) -> str:
        out = strip_invalid(normalize_markers(self._pending), self._n)
        self._pending = ""
        self.text += out
        return out

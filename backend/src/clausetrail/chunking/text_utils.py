import re

# 用后行断言切分，句末标点留在句子里
SENTENCE_SPLIT = re.compile(r"(?<=[。；;！!？?])|(?<=\n)")
LIST_LEAD_IN = re.compile(r"[:：]\s*$")
PROVISO_START = re.compile(r"^(但|但是|然而|除外|除非|不过|如果|若|则|且|并且|以及|此外|另外)")
PROVISO_END = re.compile(r"(除外|不在此限|不受此限)[。；;]?$")
LIST_ITEM = re.compile(r"^\s*([（(]?[一二三四五六七八九十\d]{1,3}[)）、.．]|[①-⑳]|[-•·])")

CJK = re.compile(r"[一-鿿]")
LATIN_WORD = re.compile(r"[A-Za-z0-9]+")


# 近似 token 数：汉字每字算 1 个，英文词和数字串按 1.3 个算，标点和空白不计
def count_tokens(text: str) -> int:
    return len(CJK.findall(text)) + int(len(LATIN_WORD.findall(text)) * 1.3)


def split_sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in SENTENCE_SPLIT.split(text)) if s]


def is_list_lead_in(text: str) -> bool:
    return bool(LIST_LEAD_IN.search(text.strip()))


def is_list_item(text: str) -> bool:
    return bool(LIST_ITEM.match(text))


def is_proviso(text: str) -> bool:
    t = text.strip()
    return bool(PROVISO_START.match(t) or PROVISO_END.search(t))


def tokenize_for_search(text: str) -> list[str]:
    from clausetrail.lexicon import tokenize

    return tokenize(text)

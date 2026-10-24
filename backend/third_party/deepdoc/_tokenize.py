import jieba
import jieba.posseg


def tokenize(text: str) -> str:
    return " ".join(t for t in jieba.lcut(text) if t.strip())


def tag(token: str) -> str:
    pairs = list(jieba.posseg.cut(token))
    return pairs[0].flag if len(pairs) == 1 else ""

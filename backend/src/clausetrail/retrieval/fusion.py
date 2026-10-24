from collections import defaultdict


# 只看各路的名次、不看原始分数，向量相似度和 BM25 分数量纲不同也能直接融合
def rrf(rankings: dict[str, list[str]], k: int = 60) -> list[tuple[str, float, dict[str, int]]]:
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for route, ids in rankings.items():
        for rank, chunk_id in enumerate(ids, start=1):
            # 同一路里重复出现的片段只按第一次（最靠前）的名次计分
            if route in ranks[chunk_id]:
                continue
            scores[chunk_id] += 1.0 / (k + rank)
            ranks[chunk_id][route] = rank
    # 总分相同时按单路最好名次排，保证顺序稳定
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], min(ranks[kv[0]].values())))
    return [(cid, score, ranks[cid]) for cid, score in ordered]

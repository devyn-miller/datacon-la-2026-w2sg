"""Stage 3: convert raw scores to ranks, then RRF to soft targets."""
import numpy as np


def within_query_ranks(scores: dict[str, float]) -> dict[str, int]:
    ranked = sorted(scores, key=scores.get, reverse=True)
    return {d: r + 1 for r, d in enumerate(ranked)}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class RankNormalized:
    def __init__(self, target: dict[str, dict[str, float]], ranks: dict[str, dict[str, dict[str, int]]]):
        self.target = target  # qid -> {doc_id: soft target probability}
        self.ranks = ranks  # qid -> source -> {doc_id: rank}


def rank_normalize(pool_by_query: dict[str, list[str]], weak_sources: dict[str, dict[str, dict[str, float]]], cfg: dict) -> RankNormalized:
    rn_cfg = cfg["rank_normalize"]
    rrf_k = rn_cfg["rrf_k"]
    temperature = rn_cfg["temperature"]

    target, ranks = {}, {}
    for qid, doc_ids in pool_by_query.items():
        ranks_by_source: dict[str, dict[str, int]] = {}
        rrf = {d: 0.0 for d in doc_ids}
        for source_name, per_query_scores in weak_sources.items():
            scores = per_query_scores.get(qid, {})
            if not scores:
                continue
            source_ranks = within_query_ranks(scores)
            ranks_by_source[source_name] = source_ranks
            for d, r in source_ranks.items():
                rrf[d] += 1.0 / (rrf_k + r)
        ranks[qid] = ranks_by_source

        rrf_vec = np.array([rrf[d] for d in doc_ids])
        if temperature <= 0:
            raise ValueError("rank_normalize.temperature must be > 0")
        probs = _softmax(rrf_vec / temperature)
        target[qid] = dict(zip(doc_ids, probs.tolist()))

    return RankNormalized(target, ranks)

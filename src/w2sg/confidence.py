"""Stage 4: label-free reliability signals (agreement/margin/entropy) + gating."""
import numpy as np

from w2sg.rank_normalize import RankNormalized


class Confidence:
    def __init__(self, weight: dict[str, dict[str, float]], keep: dict[str, dict[str, bool]], n_sources: dict[str, int]):
        self.weight = weight  # qid -> {doc_id: reliability in [0, 1]}
        self.keep = keep  # qid -> {doc_id: bool} — gate mask
        self.n_sources = n_sources  # qid -> number of sources that scored this query at all


def _agreement_votes(doc_ids: list[str], ranks_by_source: dict[str, dict[str, int]], top_k: int) -> tuple[dict[str, int], int]:
    active_sources = [s for s, r in ranks_by_source.items() if r]
    votes = {d: 0 for d in doc_ids}
    for ranks in ranks_by_source.values():
        for d, r in ranks.items():
            if r <= top_k:
                votes[d] = votes.get(d, 0) + 1
    return votes, len(active_sources)


def _normalized_margins(doc_ids: list[str], source_scores: dict[str, dict[str, float]]) -> dict[str, list[float]]:
    """Gap to next candidate, normalized by source's score range."""
    per_doc_margins: dict[str, list[float]] = {d: [] for d in doc_ids}
    for scores in source_scores.values():
        if len(scores) < 2:
            continue
        ranked = sorted(scores, key=scores.get, reverse=True)
        values = [scores[d] for d in ranked]
        score_range = max(values) - min(values)
        if score_range <= 0:
            continue
        for i, d in enumerate(ranked):
            gap = values[i] - values[i + 1] if i + 1 < len(ranked) else values[i] - values[-1]
            per_doc_margins[d].append(gap / score_range)
    return per_doc_margins


def _entropy_confidence(source_scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """1 - normalized entropy. Flat distribution = low confidence."""
    confidence_by_source = {}
    for source, scores in source_scores.items():
        if len(scores) < 2:
            confidence_by_source[source] = 0.0
            continue
        values = np.array(list(scores.values()))
        p = np.exp(values - values.max())
        p = p / p.sum()
        entropy = -(p * np.log(p + 1e-12)).sum()
        max_entropy = np.log(len(values))
        confidence_by_source[source] = 1.0 - float(entropy / max_entropy) if max_entropy > 0 else 0.0
    return confidence_by_source


def compute_confidence(
    pool_by_query: dict[str, list[str]],
    weak_sources: dict[str, dict[str, dict[str, float]]],
    rn: RankNormalized,
    cfg: dict,
) -> Confidence:
    c_cfg = cfg["confidence"]
    weight, keep, n_sources = {}, {}, {}

    for qid, doc_ids in pool_by_query.items():
        source_scores_for_query = {s: per_q.get(qid, {}) for s, per_q in weak_sources.items()}
        source_scores_for_query = {s: sc for s, sc in source_scores_for_query.items() if sc}
        n_sources[qid] = len(source_scores_for_query)

        votes, n_active = _agreement_votes(doc_ids, rn.ranks.get(qid, {}), c_cfg["agreement_top_k"])
        margins = _normalized_margins(doc_ids, source_scores_for_query)
        source_entropy_conf = _entropy_confidence(source_scores_for_query)

        q_weight, q_keep = {}, {}
        for d in doc_ids:
            doc_margins = margins.get(d, [])
            mean_margin = float(np.mean(doc_margins)) if doc_margins else 0.0
            n_agreeing = votes.get(d, 0)
            agreement_frac = n_agreeing / n_active if n_active else 0.0

            q_keep[d] = mean_margin > c_cfg["min_margin"] and n_agreeing >= c_cfg["min_agreeing_sources"]

            covering_sources = [s for s, sc in source_scores_for_query.items() if d in sc]
            entropy_term = float(np.mean([source_entropy_conf.get(s, 0.0) for s in covering_sources])) if covering_sources else 0.0
            margin_score = min(mean_margin / max(c_cfg["min_margin"] * 5, 1e-6), 1.0)
            q_weight[d] = (
                c_cfg["w_agreement"] * agreement_frac
                + c_cfg["w_margin"] * margin_score
                + c_cfg["w_entropy"] * entropy_term
            )

        weight[qid] = q_weight
        keep[qid] = q_keep

    return Confidence(weight, keep, n_sources)

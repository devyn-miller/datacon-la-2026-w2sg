"""Stage 6: evaluation (nDCG, agreement, disagreement precision, ECE, PGR)."""
import numpy as np
import torch

from w2sg.data import BeirSplit, doc_text
from w2sg.rank_normalize import RankNormalized


def ndcg_at_k(ranked_doc_ids: list[str], gold: dict[str, int], k: int) -> float:
    gains = [gold.get(d, 0) for d in ranked_doc_ids[:k]]
    dcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mean_ndcg(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int, query_ids: list[str] | None = None) -> float:
    query_ids = query_ids or list(rankings.keys())
    scores = [ndcg_at_k(rankings[q], qrels.get(q, {}), k) for q in query_ids if q in rankings and qrels.get(q)]
    return float(np.mean(scores)) if scores else float("nan")


@torch.no_grad()
def score_pool_with_model(model, split: BeirSplit, pool_by_query: dict[str, list[str]], query_ids: list[str], batch_size: int = 32) -> dict[str, dict[str, float]]:
    model.eval()
    device = next(model.parameters()).device
    out = {}
    for qid in query_ids:
        docs = pool_by_query.get(qid, [])
        if not docs:
            continue
        pairs = [(split.queries[qid], doc_text(split.corpus, d)) for d in docs]
        scores = []
        for i in range(0, len(pairs), batch_size):
            enc = model.encode_pairs(pairs[i : i + batch_size])
            enc = {k: v.to(device) for k, v in enc.items()}
            scores.extend(model(**enc).cpu().tolist())
        out[qid] = dict(zip(docs, scores))
    return out


def rank_from_scores(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=scores.get, reverse=True)


def agreement_at_k(student_scores: dict[str, float], weak_target: dict[str, float], k: int) -> float:
    student_top = set(rank_from_scores(student_scores)[:k])
    weak_top = set(rank_from_scores(weak_target)[:k])
    if not weak_top:
        return float("nan")
    return len(student_top & weak_top) / len(weak_top)


def disagreement_precision(student_scores: dict[str, float], weak_scores: dict[str, float], gold: dict[str, int]) -> tuple[int, int]:
    """Returns (correct, total) disagreeing pairs for one query."""
    docs = [d for d in student_scores if d in weak_scores]
    if len(docs) < 2:
        return 0, 0
    s = np.array([student_scores[d] for d in docs])
    w = np.array([weak_scores[d] for d in docs])
    g = np.array([gold.get(d, 0) for d in docs])

    iu = np.triu_indices(len(docs), k=1)
    s_order = np.sign(s[:, None] - s[None, :])[iu]
    w_order = np.sign(w[:, None] - w[None, :])[iu]
    g_order = np.sign(g[:, None] - g[None, :])[iu]

    disagree = (s_order != w_order) & (s_order != 0) & (w_order != 0)
    decidable = disagree & (g_order != 0)
    correct = decidable & (s_order == g_order)
    return int(correct.sum()), int(decidable.sum())


def expected_calibration_error(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Binned ECE: |accuracy - confidence| per bin."""
    if len(confidences) == 0:
        return float("nan")
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, bin_edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_idx == b
        if not in_bin.any():
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc = labels[in_bin].mean()
        ece += (in_bin.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return float(ece)


def pgr(student_ndcg: float, weak_ndcg: float, ceiling_ndcg: float) -> float:
    denom = ceiling_ndcg - weak_ndcg
    if abs(denom) < 1e-9:
        return float("nan")
    return (student_ndcg - weak_ndcg) / denom


def evaluate_model(
    model,
    split: BeirSplit,
    pool_by_query: dict[str, list[str]],
    eval_rn: RankNormalized,
    cfg: dict,
    query_ids: list[str] | None = None,
) -> dict:
    """Returns nDCG, agreement, disagreement precision, ECE."""
    e_cfg = cfg["eval"]
    query_ids = query_ids or list(pool_by_query.keys())
    max_q = cfg["train"].get("max_eval_queries")
    if max_q:
        query_ids = query_ids[:max_q]

    student_scores = score_pool_with_model(model, split, pool_by_query, query_ids)

    rankings = {q: rank_from_scores(s) for q, s in student_scores.items()}
    gold_ndcg = mean_ndcg(rankings, split.qrels, e_cfg["ndcg_k"], query_ids)

    agree_vals, correct_total, decidable_total = [], 0, 0
    confidences, labels = [], []
    for qid in query_ids:
        weak_target = eval_rn.target.get(qid)
        if not weak_target or qid not in student_scores:
            continue
        agree_vals.append(agreement_at_k(student_scores[qid], weak_target, e_cfg["agreement_k"]))
        c, d = disagreement_precision(student_scores[qid], weak_target, split.qrels.get(qid, {}))
        correct_total += c
        decidable_total += d

        for doc_id, logit in student_scores[qid].items():
            confidences.append(1 / (1 + np.exp(-logit)))
            labels.append(1 if split.qrels.get(qid, {}).get(doc_id, 0) > 0 else 0)

    return {
        "ndcg": gold_ndcg,
        "agreement_at_k": float(np.nanmean(agree_vals)) if agree_vals else float("nan"),
        "disagreement_precision": correct_total / decidable_total if decidable_total else float("nan"),
        "n_disagreeing_pairs": decidable_total,
        "ece": expected_calibration_error(np.array(confidences), np.array(labels), e_cfg["ece_bins"]),
    }

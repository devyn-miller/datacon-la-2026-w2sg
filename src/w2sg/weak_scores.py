"""Stage 2: score candidates with weak sources (BM25, dense, small CE, debiased CTR)."""
import logging

import numpy as np

from w2sg.candidates import CandidatePool
from w2sg.data import BeirSplit, doc_text
from w2sg.models import load_cross_encoder_scorer

logger = logging.getLogger(__name__)


class WeakScores:
    def __init__(self, sources: dict[str, dict[str, dict[str, float]]]):
        self.sources = sources

    def names(self) -> list[str]:
        return list(self.sources.keys())


def _relevance_to_click_prob(rel: int) -> float:
    return min(0.05 + 0.45 * rel, 0.95)


def _simulate_debiased_ctr(query_ids, pool, split, bm25_ranks_by_query, cfg, rng) -> dict[str, dict[str, float]]:
    eta = cfg["position_bias_eta"]
    n_sessions = cfg["n_sessions_per_query"]
    noise_std = cfg["click_noise_std"]
    out = {}
    for qid in query_ids:
        gold = split.qrels.get(qid, {})
        ranks = bm25_ranks_by_query[qid]
        scores = {}
        for doc_id in pool.pools[qid]:
            if doc_id not in ranks:
                continue
            position = ranks[doc_id]
            examine_prob = position ** (-eta)
            rel = gold.get(doc_id, 0)
            click_prob = np.clip(_relevance_to_click_prob(rel) + rng.normal(0, noise_std), 0.0, 1.0)
            clicks = rng.binomial(n_sessions, examine_prob * click_prob)
            raw_ctr = clicks / n_sessions
            scores[doc_id] = raw_ctr / examine_prob
        out[qid] = scores
    return out


def score_weak_sources(split: BeirSplit, pool: CandidatePool, cfg: dict, use_ctr: bool) -> WeakScores:
    ws_cfg = cfg["weak_sources"]
    query_ids = list(pool.pools.keys())
    sources: dict[str, dict[str, dict[str, float]]] = {}

    bm25_ranks_by_query = {}  # qid -> {doc_id: 1-indexed rank}, used by both BM25 scoring and CTR simulation

    if ws_cfg.get("bm25", True):
        bm25_scores = {}
        for qid in query_ids:
            tokenized = split.queries[qid].lower().split()
            raw = pool.bm25_index.get_scores(tokenized)
            doc_score = {d: raw[i] for i, d in enumerate(pool.bm25_doc_ids)}
            candidate_scores = {d: doc_score[d] for d in pool.pools[qid] if d in doc_score}
            bm25_scores[qid] = candidate_scores
            ranked = sorted(candidate_scores, key=candidate_scores.get, reverse=True)
            bm25_ranks_by_query[qid] = {d: r + 1 for r, d in enumerate(ranked)}
        sources["bm25"] = bm25_scores

    if ws_cfg.get("dense_bi_encoder", {}).get("enabled", True):
        dense_scores = {}
        doc_emb_cache: dict[str, np.ndarray] = {}
        for qid in query_ids:
            docs = pool.pools[qid]
            missing = [d for d in docs if d not in doc_emb_cache]
            if missing:
                texts = [doc_text(split.corpus, d) for d in missing]
                embs = pool.dense_encoder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                for d, e in zip(missing, embs):
                    doc_emb_cache[d] = e
            q_emb = pool.dense_encoder.encode(split.queries[qid], normalize_embeddings=True)
            dense_scores[qid] = {d: float(doc_emb_cache[d] @ q_emb) for d in docs}
        sources["dense_bi_encoder"] = dense_scores

    ce_cfg = ws_cfg.get("small_cross_encoder", {})
    if ce_cfg.get("enabled", True):
        scorer = load_cross_encoder_scorer(ce_cfg["model_name"], ce_cfg["max_length"], offline=cfg.get("offline", False))
        top_k = ce_cfg["score_top_k"]
        ce_scores = {}
        for qid in query_ids:
            docs = pool.pools[qid]
            if qid in bm25_ranks_by_query:
                docs = sorted(docs, key=lambda d: bm25_ranks_by_query[qid].get(d, len(docs) + 1))[:top_k]
            else:
                docs = docs[:top_k]
            pairs = [(split.queries[qid], doc_text(split.corpus, d)) for d in docs]
            scores = scorer.predict(pairs)
            ce_scores[qid] = dict(zip(docs, scores))
        sources["small_cross_encoder"] = ce_scores

    ctr_cfg = ws_cfg.get("debiased_ctr", {})
    if use_ctr and ctr_cfg.get("enabled", True):
        if not bm25_ranks_by_query:
            raise RuntimeError("debiased_ctr needs BM25 ranks; enable weak_sources.bm25")
        rng = np.random.default_rng(cfg["train"]["seed"])
        sources["debiased_ctr"] = _simulate_debiased_ctr(query_ids, pool, split, bm25_ranks_by_query, ctr_cfg, rng)

    for name, per_query in sources.items():
        covered = sum(len(v) for v in per_query.values())
        total = sum(len(pool.pools[q]) for q in query_ids)
        logger.info("weak source '%s' covers %d / %d (query, candidate) pairs", name, covered, total)

    return WeakScores(sources)

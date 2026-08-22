"""Stage 1: candidate pool (BM25 + dense + random negatives, with recall check)."""
import logging
import random

import numpy as np
from rank_bm25 import BM25Okapi

from w2sg.data import BeirSplit, doc_text
from w2sg.models import load_dense_encoder

logger = logging.getLogger(__name__)


class CandidatePool:
    def __init__(self, pools: dict[str, list[str]], recall_at_k: dict, bm25_index, bm25_doc_ids, dense_encoder):
        self.pools = pools
        self.recall = recall_at_k
        self.bm25_index = bm25_index
        self.bm25_doc_ids = bm25_doc_ids
        self.dense_encoder = dense_encoder


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index(corpus: dict) -> tuple[BM25Okapi, list[str]]:
    doc_ids = list(corpus.keys())
    tokenized = [_tokenize(doc_text(corpus, d)) for d in doc_ids]
    return BM25Okapi(tokenized), doc_ids


def build_candidate_pool(split: BeirSplit, cfg: dict, query_ids: list[str] | None = None) -> CandidatePool:
    cand_cfg = cfg["candidates"]
    query_ids = query_ids or list(split.queries.keys())
    corpus_doc_ids = list(split.corpus.keys())

    bm25, bm25_doc_ids = build_bm25_index(split.corpus)

    dense = load_dense_encoder(cand_cfg["dense_encoder"], offline=cfg.get("offline", False))
    corpus_texts = [doc_text(split.corpus, d) for d in corpus_doc_ids]
    corpus_emb = dense.encode(corpus_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    rng = random.Random(cfg["train"]["seed"])
    pools: dict[str, list[str]] = {}
    recall_hits, recall_total = 0, 0
    per_query_recall = {}

    for qid in query_ids:
        query_text = split.queries[qid]

        bm25_scores = bm25.get_scores(_tokenize(query_text))
        bm25_top = np.argsort(-bm25_scores)[: cand_cfg["bm25_top_k"]]
        bm25_ids = [bm25_doc_ids[i] for i in bm25_top]

        q_emb = dense.encode(query_text, normalize_embeddings=True)
        dense_scores = corpus_emb @ q_emb
        dense_top = np.argsort(-dense_scores)[: cand_cfg["dense_top_k"]]
        dense_ids = [corpus_doc_ids[i] for i in dense_top]

        pool = list(dict.fromkeys(bm25_ids + dense_ids))
        n_random = cand_cfg["n_random_negatives"]
        retriever_budget = cand_cfg["max_pool_size"] - (n_random if n_random else 0)
        if len(pool) > retriever_budget:
            pool = pool[:retriever_budget]

        if n_random:
            remaining = [d for d in corpus_doc_ids if d not in pool]
            pool += rng.sample(remaining, min(n_random, len(remaining)))

        pools[qid] = pool

        gold_relevant = {d for d, rel in split.qrels.get(qid, {}).items() if rel > 0}
        if gold_relevant:
            hits = len(gold_relevant & set(pool))
            per_query_recall[qid] = hits / len(gold_relevant)
            recall_hits += hits
            recall_total += len(gold_relevant)

    recall_at_k = {
        "micro": recall_hits / recall_total if recall_total else float("nan"),
        "macro": float(np.mean(list(per_query_recall.values()))) if per_query_recall else float("nan"),
        "per_query": per_query_recall,
        "k": cand_cfg["recall_k"],
        "n_queries_with_gold": len(per_query_recall),
    }

    logger.info(
        "Recall@%d vs gold: micro=%.3f macro=%.3f (%d queries with judgments)",
        recall_at_k["k"], recall_at_k["micro"], recall_at_k["macro"], recall_at_k["n_queries_with_gold"],
    )
    min_recall = cand_cfg["min_acceptable_recall"]
    if recall_at_k["macro"] < min_recall:
        raise RuntimeError(f"Recall@{recall_at_k['k']} = {recall_at_k['macro']:.3f} < {min_recall}")

    return CandidatePool(pools, recall_at_k, bm25, bm25_doc_ids, dense)

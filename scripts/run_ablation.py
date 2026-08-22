#!/usr/bin/env python3
"""Runs full ablation table: BM25 -> naive -> +gating -> +confidence -> +self_term -> gold."""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from w2sg.candidates import build_candidate_pool
from w2sg.config import load_config
from w2sg.confidence import compute_confidence
from w2sg.data import load_dataset
from w2sg.evaluate import evaluate_model, mean_ndcg, pgr, rank_from_scores, score_pool_with_model
from w2sg.rank_normalize import rank_normalize
from w2sg.train import CONFIDENCE, FULL, GATING, GOLD_CEILING, NAIVE, train_student
from w2sg.weak_scores import score_weak_sources

from eval.failure_modes import (
    check_agreement_never_declines,
    check_pool_recall,
    check_self_reinforcing_collapse,
    collapse_correlated_sources,
    flag_correlated_sources,
    stratified_ndcg,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

VARIANTS = [
    ("naive_ft", NAIVE),
    ("+gating", GATING),
    ("+confidence", CONFIDENCE),
    ("+self_term", FULL),
    ("gold_ceiling", GOLD_CEILING),
]


def make_eval_callback(cfg, eval_split, eval_pool, eval_rn, history):
    def on_eval(step, model):
        metrics = evaluate_model(model, eval_split, eval_pool, eval_rn, cfg)
        history.append({**metrics, "step": step})
        collapse = check_self_reinforcing_collapse(
            history, cfg["failure_modes"]["collapse_ndcg_drop_threshold"], cfg["failure_modes"]["collapse_agreement_drop_threshold"]
        )
        if collapse["triggered"]:
            metrics["abort"] = True
            metrics["abort_reason"] = collapse["reason"]
        return metrics

    return on_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    os.makedirs(args.out_dir, exist_ok=True)

    ds_cfg = cfg["dataset"]
    if ds_cfg["name"] == "synthetic_demo" and not os.path.exists(os.path.join(ds_cfg["data_dir"], "synthetic_demo")):
        raise RuntimeError("run scripts/make_synthetic_beir.py first to generate the offline fixture")

    train_split, eval_split = load_dataset(ds_cfg["name"], ds_cfg["data_dir"], ds_cfg["train_split"], ds_cfg["eval_split"])

    train_qids = list(train_split.queries.keys())
    eval_qids = list(eval_split.queries.keys())
    max_train_q, max_eval_q = cfg["train"].get("max_train_queries"), cfg["train"].get("max_eval_queries")
    if max_train_q:
        train_qids = train_qids[:max_train_q]
    if max_eval_q:
        eval_qids = eval_qids[:max_eval_q]

    logger.info("=== stage 1: candidates ===")
    train_pool = build_candidate_pool(train_split, cfg, query_ids=train_qids)
    eval_pool = build_candidate_pool(eval_split, cfg, query_ids=eval_qids)

    logger.info("=== stage 2: weak scores ===")
    train_weak = score_weak_sources(train_split, train_pool, cfg, use_ctr=True)
    eval_weak = score_weak_sources(eval_split, eval_pool, cfg, use_ctr=False)

    correlated = flag_correlated_sources(train_weak.sources, cfg["confidence"]["kendall_tau_correlation_threshold"])
    if correlated:
        for a, b, tau in correlated:
            logger.warning("weak sources '%s' and '%s' correlated at tau=%.2f, collapsing", a, b, tau)
        train_weak.sources, dropped = collapse_correlated_sources(train_weak.sources, cfg["confidence"]["kendall_tau_correlation_threshold"])
        for name in dropped:
            eval_weak.sources.pop(name, None)

    logger.info("=== stage 3: rank-normalize ===")
    train_rn = rank_normalize(train_pool.pools, train_weak.sources, cfg)
    eval_rn = rank_normalize(eval_pool.pools, eval_weak.sources, cfg)

    logger.info("=== stage 4: confidence ===")
    train_conf = compute_confidence(train_pool.pools, train_weak.sources, train_rn, cfg)
    n_gated = sum(1 for q in train_conf.keep for d, k in train_conf.keep[q].items() if not k)
    n_total = sum(len(v) for v in train_conf.keep.values())
    logger.info("gating removed %d / %d (query, candidate) pairs (%.1f%%)", n_gated, n_total, 100 * n_gated / n_total if n_total else 0)

    rows = {}

    logger.info("=== row: BM25 (untrained baseline) ===")
    bm25_rankings = {q: rank_from_scores(s) for q, s in eval_weak.sources.get("bm25", {}).items()}
    rows["BM25"] = {"ndcg": mean_ndcg(bm25_rankings, eval_split.qrels, cfg["eval"]["ndcg_k"], eval_qids)}

    trained_models = {}
    for name, flags in VARIANTS:
        logger.info("=== row: %s ===", name)
        history = []
        callback = make_eval_callback(cfg, eval_split, eval_pool.pools, eval_rn, history)
        model, run_history = train_student(
            cfg, train_split, train_pool.pools, train_rn, train_conf, flags, query_ids=train_qids, on_eval=callback
        )
        trained_models[name] = model
        final_metrics = evaluate_model(model, eval_split, eval_pool.pools, eval_rn, cfg, query_ids=eval_qids)
        rows[name] = final_metrics
        rows[name]["_history"] = run_history

        if name == "naive_ft":
            never_declines = check_agreement_never_declines(run_history, cfg["failure_modes"]["agreement_never_declines_tolerance"])
            logger.info("naive_ft agreement-never-declines check: %s", never_declines["reason"])

    weak_ndcg = rows["BM25"]["ndcg"]
    ceiling_ndcg = rows["gold_ceiling"]["ndcg"]
    for name, _ in VARIANTS:
        rows[name]["pgr"] = pgr(rows[name]["ndcg"], weak_ndcg, ceiling_ndcg)

    full_scores = score_pool_with_model(trained_models["+self_term"], eval_split, eval_pool.pools, eval_qids)
    full_rankings = {q: rank_from_scores(s) for q, s in full_scores.items()}
    strat = stratified_ndcg(full_rankings, eval_split.qrels, eval_qids, cfg["eval"]["ndcg_k"])

    recall_check = check_pool_recall(eval_pool.recall, cfg["candidates"]["min_acceptable_recall"])

    report = {
        "config": args.config,
        "override": args.override,
        "recall_at_100": eval_pool.recall,
        "recall_check": recall_check,
        "correlated_sources": correlated,
        "rows": {k: {kk: vv for kk, vv in v.items() if kk != "_history"} for k, v in rows.items()},
        "head_torso_tail_ndcg": strat,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.out_dir, f"ablation_{timestamp}.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print_table(report)
    with open(os.path.join(args.out_dir, "ablation_table.md"), "w") as f:
        f.write(render_markdown(report))


def print_table(report: dict):
    print("\n" + "=" * 78)
    print("ABLATION: BM25 -> naive FT -> +gating -> +confidence -> +self term -> gold ceiling")
    print("=" * 78)
    print(f"{'row':<14}{'nDCG@10':>10}{'PGR':>8}{'Agree@10':>10}{'DisagrPrec':>12}{'ECE':>8}")
    for name in ["BM25", "naive_ft", "+gating", "+confidence", "+self_term", "gold_ceiling"]:
        r = report["rows"][name]
        print(
            f"{name:<14}{r.get('ndcg', float('nan')):>10.3f}{r.get('pgr', float('nan')):>8.3f}"
            f"{r.get('agreement_at_k', float('nan')):>10.3f}{r.get('disagreement_precision', float('nan')):>12.3f}"
            f"{r.get('ece', float('nan')):>8.3f}"
        )
    print("=" * 78)
    print(f"Recall@100 (eval pool): {report['recall_at_100']['macro']:.3f} — {report['recall_check']['reason']}")
    if report["correlated_sources"]:
        print(f"Correlated weak sources collapsed: {report['correlated_sources']}")
    ht = report["head_torso_tail_ndcg"]
    print(f"Head/torso/tail nDCG@10 (+self_term model): head={ht['head']:.3f} torso={ht['torso']:.3f} tail={ht['tail']:.3f}")


def render_markdown(report: dict) -> str:
    lines = [
        "# Ablation results",
        "",
        f"Config: `{report['config']}`" + (f" + `{report['override']}`" if report["override"] else ""),
        "",
        f"Recall@100 (eval pool): **{report['recall_at_100']['macro']:.3f}** — {report['recall_check']['reason']}",
        "",
        "| row | nDCG@10 | PGR | Agreement@10 | Disagreement precision | ECE |",
        "|---|---|---|---|---|---|",
    ]
    for name in ["BM25", "naive_ft", "+gating", "+confidence", "+self_term", "gold_ceiling"]:
        r = report["rows"][name]
        lines.append(
            f"| {name} | {r.get('ndcg', float('nan')):.3f} | {r.get('pgr', float('nan')):.3f} | "
            f"{r.get('agreement_at_k', float('nan')):.3f} | {r.get('disagreement_precision', float('nan')):.3f} | "
            f"{r.get('ece', float('nan')):.3f} |"
        )
    ht = report["head_torso_tail_ndcg"]
    lines += [
        "",
        f"Head/torso/tail nDCG@10 (+self_term model): head={ht['head']:.3f} (n={ht['n_head']}), "
        f"torso={ht['torso']:.3f} (n={ht['n_torso']}), tail={ht['tail']:.3f} (n={ht['n_tail']})",
    ]
    if report["correlated_sources"]:
        lines += ["", f"Correlated weak sources flagged and collapsed: {report['correlated_sources']}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

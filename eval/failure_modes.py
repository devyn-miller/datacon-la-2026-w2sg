"""Training diagnostics that return actionable verdicts."""
import hashlib

import numpy as np
from scipy.stats import kendalltau

from w2sg.evaluate import mean_ndcg


def check_agreement_never_declines(history: list[dict], tolerance: float) -> dict:
    """Checks if alpha had any effect (benign if triggered)."""
    if len(history) < 2:
        return {"triggered": False, "reason": "not enough eval steps yet"}
    early = history[0]["agreement_at_k"]
    late = history[-1]["agreement_at_k"]
    declined = (early - late) > tolerance
    if declined:
        return {"triggered": False, "reason": f"agreement dropped {early:.3f} -> {late:.3f}, ramp is doing something"}
    return {
        "triggered": True,
        "reason": f"agreement@k stayed flat ({early:.3f} -> {late:.3f}); raise alpha_ceiling or move the ramp earlier",
    }


def check_self_reinforcing_collapse(history: list[dict], ndcg_drop_threshold: float, agreement_drop_threshold: float) -> dict:
    """Detects self-term locked onto early noise. Abort if triggered."""
    if len(history) < 3:
        return {"triggered": False, "reason": "not enough eval steps yet"}
    ndcgs = [h["ndcg"] for h in history]
    agreements = [h["agreement_at_k"] for h in history]
    peak_ndcg, peak_idx = max((v, i) for i, v in enumerate(ndcgs))
    current_ndcg, current_agreement = ndcgs[-1], agreements[-1]
    alpha_now = history[-1].get("alpha", 0.0)

    ndcg_drop = peak_ndcg - current_ndcg
    agreement_drop = agreements[peak_idx] - current_agreement if peak_idx < len(agreements) else 0.0

    collapsed = alpha_now > 0.2 and ndcg_drop > ndcg_drop_threshold and agreement_drop > agreement_drop_threshold
    if collapsed:
        return {
            "triggered": True,
            "reason": (
                f"nDCG dropped {ndcg_drop:.3f} and agreement dropped {agreement_drop:.3f} from their peaks "
                f"at alpha={alpha_now:.2f}; self-term locked onto early noise. Restart with a longer warmup, "
                f"lower alpha_ceiling, or higher self_term_tau; this does not recover mid-run."
            ),
        }
    return {"triggered": False, "reason": "no collapse signature"}


def kendall_tau_matrix(weak_sources: dict[str, dict[str, dict[str, float]]]) -> dict[tuple[str, str], float]:
    """Finds correlated sources (tau > ~0.9 = same vote counted twice)."""
    names = list(weak_sources.keys())
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            taus = []
            for qid in weak_sources[a_name]:
                a, b = weak_sources[a_name][qid], weak_sources[b_name].get(qid, {})
                shared = [d for d in a if d in b]
                if len(shared) < 3:
                    continue
                tau, _ = kendalltau([a[d] for d in shared], [b[d] for d in shared])
                if not np.isnan(tau):
                    taus.append(tau)
            if taus:
                out[(a_name, b_name)] = float(np.mean(taus))
    return out


def flag_correlated_sources(weak_sources: dict, threshold: float) -> list[tuple[str, str, float]]:
    return [(a, b, tau) for (a, b), tau in kendall_tau_matrix(weak_sources).items() if tau > threshold]


def collapse_correlated_sources(weak_sources: dict, threshold: float) -> tuple[dict, list[str]]:
    """Drops all but one from each correlated pair (keeps first alphabetically)."""
    flagged = flag_correlated_sources(weak_sources, threshold)
    dropped = set()
    for a, b, _ in flagged:
        if a not in dropped and b not in dropped:
            dropped.add(b)
    remaining = {name: scores for name, scores in weak_sources.items() if name not in dropped}
    return remaining, sorted(dropped)


def check_pool_recall(recall_report: dict, min_acceptable: float) -> dict:
    """Validates recall floor (stage 1 already checks, this is for final report)."""
    macro = recall_report.get("macro", float("nan"))
    if np.isnan(macro) or macro < min_acceptable:
        return {
            "triggered": True,
            "reason": f"Recall@{recall_report.get('k')} = {macro:.3f} is below {min_acceptable}. "
            f"Every nDCG number below is upper-bounded by this; treat them as suspect, not just low.",
        }
    return {"triggered": False, "reason": f"Recall@{recall_report.get('k')} = {macro:.3f}, above floor"}


def synthetic_query_frequency(query_ids: list[str], n_deciles: int = 10, seed: int = 42) -> dict[str, int]:
    """Hash-based Zipfian query frequency (BEIR has no real traffic data)."""
    weights = {}
    for qid in query_ids:
        h = int(hashlib.sha1(f"{seed}:{qid}".encode()).hexdigest()[:8], 16)
        weights[qid] = (h / 0xFFFFFFFF) ** 2

    ranked = sorted(query_ids, key=lambda q: weights[q], reverse=True)  # highest weight = most "frequent" = head
    decile = {}
    per_bucket = max(1, len(ranked) // n_deciles)
    for i, qid in enumerate(ranked):
        decile[qid] = min(i // per_bucket, n_deciles - 1)
    return decile


def stratified_ndcg(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], query_ids: list[str], k: int, n_deciles: int = 10, seed: int = 42) -> dict:
    """Reports head/torso/tail nDCG separately to detect head-only gains."""
    decile = synthetic_query_frequency(query_ids, n_deciles, seed)
    head = [q for q in query_ids if decile.get(q, 0) < n_deciles * 0.3]
    torso = [q for q in query_ids if n_deciles * 0.3 <= decile.get(q, 0) < n_deciles * 0.7]
    tail = [q for q in query_ids if decile.get(q, 0) >= n_deciles * 0.7]
    return {
        "head": mean_ndcg(rankings, qrels, k, head),
        "torso": mean_ndcg(rankings, qrels, k, torso),
        "tail": mean_ndcg(rankings, qrels, k, tail),
        "n_head": len(head),
        "n_torso": len(torso),
        "n_tail": len(tail),
    }

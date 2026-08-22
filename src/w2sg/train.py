"""Stage 5: training loop and ablation variants."""
import logging
import random

import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from w2sg.confidence import Confidence
from w2sg.data import BeirSplit, doc_text
from w2sg.loss import alpha_schedule, w2s_listwise_loss
from w2sg.models import load_student
from w2sg.rank_normalize import RankNormalized

logger = logging.getLogger(__name__)


class VariantFlags:
    """Ablation row feature flags."""

    def __init__(self, use_gate: bool, use_confidence_weight: bool, use_self_term: bool, use_gold_target: bool):
        self.use_gate = use_gate
        self.use_confidence_weight = use_confidence_weight
        self.use_self_term = use_self_term
        self.use_gold_target = use_gold_target


NAIVE = VariantFlags(use_gate=False, use_confidence_weight=False, use_self_term=False, use_gold_target=False)
GATING = VariantFlags(use_gate=True, use_confidence_weight=False, use_self_term=False, use_gold_target=False)
CONFIDENCE = VariantFlags(use_gate=True, use_confidence_weight=True, use_self_term=False, use_gold_target=False)
FULL = VariantFlags(use_gate=True, use_confidence_weight=True, use_self_term=True, use_gold_target=False)
GOLD_CEILING = VariantFlags(use_gate=False, use_confidence_weight=False, use_self_term=False, use_gold_target=True)


def _gold_target(doc_ids: list[str], qrels_for_query: dict[str, int]) -> list[float]:
    grades = np.array([max(qrels_for_query.get(d, 0), 0) for d in doc_ids], dtype=float)
    if grades.sum() > 0:
        return (grades / grades.sum()).tolist()
    return (np.ones(len(doc_ids)) / len(doc_ids)).tolist()


class TrainingExample:
    def __init__(self, qid: str, doc_ids: list[str], target: list[float], conf: list[float]):
        self.qid = qid
        self.doc_ids = doc_ids
        self.target = target
        self.conf = conf


def build_training_examples(
    query_ids: list[str],
    pool_by_query: dict[str, list[str]],
    rn: RankNormalized | None,
    confidence: Confidence | None,
    flags: VariantFlags,
    list_size: int,
    qrels: dict[str, dict[str, int]] | None = None,
) -> list[TrainingExample]:
    examples = []
    for qid in query_ids:
        pool = pool_by_query.get(qid, [])
        if len(pool) < 2:
            continue

        if flags.use_gold_target:
            if not qrels or qid not in qrels or not any(v > 0 for v in qrels[qid].values()):
                continue
            positives = [d for d in pool if qrels[qid].get(d, 0) > 0]
            negatives = [d for d in pool if d not in positives]
            if rn is not None and qid in rn.target:
                negatives = sorted(negatives, key=lambda d: rn.target[qid].get(d, 0.0), reverse=True)
            doc_ids = (positives + negatives)[:list_size]
            target = _gold_target(doc_ids, qrels[qid])
            conf = [1.0] * len(doc_ids)
        else:
            target_map = rn.target[qid]
            ranked = sorted(pool, key=lambda d: target_map.get(d, 0.0), reverse=True)
            doc_ids = ranked[:list_size]
            raw_target = np.array([target_map[d] for d in doc_ids])

            if flags.use_gate:
                keep_mask = np.array([confidence.keep[qid].get(d, True) for d in doc_ids])
                if not keep_mask.any():
                    continue
                raw_target = raw_target * keep_mask
            raw_target = raw_target / raw_target.sum()
            target = raw_target.tolist()

            if flags.use_confidence_weight:
                conf = [confidence.weight[qid].get(d, 0.0) for d in doc_ids]
                if flags.use_gate:
                    conf = [c * float(confidence.keep[qid].get(d, True)) for c, d in zip(conf, doc_ids)]
            elif flags.use_gate:
                conf = [float(confidence.keep[qid].get(d, True)) for d in doc_ids]
            else:
                conf = [1.0] * len(doc_ids)

        examples.append(TrainingExample(qid, doc_ids, target, conf))
    return examples


def _make_batch(examples: list[TrainingExample], list_size: int, model, corpus: dict, queries: dict, device):
    B = len(examples)
    pairs, target = [], torch.zeros(B, list_size)
    conf = torch.zeros(B, list_size)
    mask = torch.zeros(B, list_size, dtype=torch.bool)

    for i, ex in enumerate(examples):
        n = len(ex.doc_ids)
        q_text = queries[ex.qid]
        for j in range(list_size):
            if j < n:
                pairs.append((q_text, doc_text(corpus, ex.doc_ids[j])))
                target[i, j] = ex.target[j]
                conf[i, j] = ex.conf[j]
                mask[i, j] = True
            else:
                pairs.append((q_text, ""))

    encoded = model.encode_pairs(pairs)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    return encoded, target.to(device), conf.to(device), mask.to(device)


def train_student(
    cfg: dict,
    split: BeirSplit,
    pool_by_query: dict[str, list[str]],
    rn: RankNormalized | None,
    confidence: Confidence | None,
    flags: VariantFlags,
    query_ids: list[str] | None = None,
    on_eval=None,
):
    """Trains one ablation variant. on_eval can trigger early abort."""
    t_cfg = cfg["train"]
    seed = t_cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_student(cfg["model"]["student"], cfg["model"]["max_length"], offline=cfg.get("offline", False))
    model.to(device)

    query_ids = query_ids or list(pool_by_query.keys())
    max_q = t_cfg.get("max_train_queries")
    if max_q:
        query_ids = query_ids[:max_q]

    examples = build_training_examples(
        query_ids, pool_by_query, rn, confidence, flags, t_cfg["list_size"], qrels=split.qrels if flags.use_gold_target else None
    )
    if not examples:
        raise RuntimeError("No training examples survived gating")
    logger.info("training on %d / %d queries after filtering", len(examples), len(query_ids))

    batch_size = t_cfg["batch_size"]
    n_batches_per_epoch = (len(examples) + batch_size - 1) // batch_size
    total_steps = n_batches_per_epoch * t_cfg["epochs"]

    optimizer = AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * t_cfg["warmup_frac"]), num_training_steps=total_steps
    )

    step = 0
    history = []
    rng = random.Random(seed)
    for epoch in range(t_cfg["epochs"]):
        rng.shuffle(examples)
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i : i + batch_size]
            encoded, target, conf, mask = _make_batch(batch_examples, t_cfg["list_size"], model, split.corpus, split.queries, device)

            model.train()
            logits = model(**encoded).view(len(batch_examples), t_cfg["list_size"])

            alpha = alpha_schedule(step, total_steps, t_cfg["alpha_ceiling"], t_cfg["alpha_ramp_start_frac"], t_cfg["alpha_ramp_end_frac"]) if flags.use_self_term else 0.0
            loss = w2s_listwise_loss(logits, target, conf, alpha, tau=t_cfg["self_term_tau"], mask=mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            step += 1
            if on_eval and step % t_cfg["eval_every_steps"] == 0:
                metrics = on_eval(step, model)
                metrics["alpha"] = alpha
                metrics["loss"] = float(loss.item())
                history.append(metrics)
                if metrics.get("abort"):
                    logger.warning("training aborted at step %d: %s", step, metrics.get("abort_reason"))
                    return model, history

    return model, history

Weak-to-strong generalization for reranking

Reference implementation for the DataConLA 2026 talk on training a strong
cross-encoder reranker on weak, noisy labels (BM25, a small distilled
reranker, click logs) without just teaching it to reproduce those signals'
mistakes.

Standard practice fine-tunes a reranker with cross-entropy against whatever
weak label you have. That objective's global minimum is "match the weak
label everywhere," including everywhere the weak label is wrong. Nothing in
plain CE rewards being *more* correct than the thing supervising you. This
repo implements the fix: a confidence-aware loss that adds a scheduled
self-consistency term the weak labeler doesn't control, plus the
instrumentation to prove it actually recovered capability instead of just
producing a model that agrees with itself more confidently.

Background: Burns et al., *Weak-to-Strong Generalization*
([arXiv:2312.09390](https://arxiv.org/abs/2312.09390)); the RRF fusion is
Cormack, Clarke & Buettcher, SIGIR 2009; the ranking-over-regression argument
for weak IR supervision is Dehghani et al., SIGIR 2017
([arXiv:1704.08803](https://arxiv.org/abs/1704.08803)); unbiased
click-through debiasing is Joachims et al., WSDM 2017; the dataset is BEIR
(Thakur et al., 2021).

## The six stages

Each is its own module under `src/w2sg/`, matching the boundaries the talk
walks through:

1. __`candidates.py`__ — candidate pool per query, as a union of BM25 top-k,
   dense bi-encoder top-k, and random negatives. Never just "whatever the
   weak labeler already ranked" — if the pool-builder and the labeler are
   the same system, the student can never learn about a document the
   labeler missed. Recall@100 against gold is computed and checked here
   unconditionally; it raises if it's below `candidates.min_acceptable_recall`,
   because every ranking metric downstream is upper-bounded by it.

2. __`weak_scores.py`__ — scores every pool candidate with each weak source
   independently: BM25, a dense bi-encoder (reusing stage 1's embeddings),
   `cross-encoder/ms-marco-MiniLM-L-6-v2` (top-k only; everything outside
   its budget is _missing_, not scored low), and a simulated debiased-CTR
   signal (BEIR has no click logs, so this simulates a position-biased click
   process from train-split gold relevance and then corrects for the
   position bias — see `_simulate_debiased_ctr`). The CTR source only ever
   runs on the train split; building it needs qrels, and running it against
   test qrels would leak gold into a signal the eval treats as "weak."

3. __`rank_normalize.py`__ — converts each source's raw scores to
   within-query ranks (raw scores aren't comparable across sources or
   queries), fuses them with RRF (`1/(k+rank)`, k=60, Cormack et al. 2009),
   and softmaxes the fused weight with a temperature to get a soft target
   distribution per query.

4. **`confidence.py`** — three label-free reliability signals per candidate:
   agreement (fraction of sources that place it in their own top-k), margin
   (score gap to the next-ranked candidate, normalized per-source by that
   source's own score range), and list entropy (how flat a source's score
   distribution is over the pool, so flat means that source has no opinion
   about this query). These combine into a reliability weight, and
   candidates below a hard margin/agreement floor are gated out entirely.
   Gating happens *before* the smooth weight is computed, so a gated
   candidate contributes exactly zero gradient instead of a diluted one.

5. **`loss.py`** + **`train.py`** — the loss itself:

```md
loss = (1 - alpha) * reliability_weighted_CE(weak_target, student_logits)
     + alpha * CE(sharpen(student_logits).detach(), student_logits)
```

`alpha` ramps from 0 to a configurable ceiling on a schedule that starts
only after the model has learned the basic task from the weak targets.
Turning the self-term on at step 0 makes it lock onto the randomly
initialized head's arbitrary first guesses. `train.py` builds the
listwise training batches and implements the five ablation variants as
flag combinations (`VariantFlags` in `train.py`), not five copies of the
training loop.

6. **`evaluate.py`** — the four numbers that matter, computed every eval
   step: **nDCG@10** against gold (the only unbiased signal anywhere in this
   pipeline), **Agreement@10** with the weak labeler (should decline from
   fine-tuning, then plateau — if it never declines, alpha's ceiling is too
   low or the ramp starts too late), **disagreement precision** (when the
   student and the weak labeler disagree on a pair's relative order, how
   often is the student's order actually right against gold, so if below 0.5,
   back off alpha), and **ECE** (ties `sigmoid(logit)` to actual binary gold
   relevance; if this is bad, the self-term is amplifying the student's own
   miscalibration instead of correcting anything).

## Failure-mode diagnostics (`eval/failure_modes.py`)

Five runnable checks, not just numbers to eyeball:

- __Agreement never declines__ — benign; means you've reproduced naive
   fine-tuning, raise `alpha_ceiling` or move the ramp earlier.
- __Self-reinforcing collapse__ — nDCG and agreement both drop sharply once
   alpha crosses a threshold. This doesn't recover mid-run, so
   `check_self_reinforcing_collapse` is wired into `train.py`'s eval callback
   and aborts the run early rather than burning the rest of the compute
   budget on a run that's already lost.
- __Correlated weak sources__ — Kendall's tau between every pair of weak
   sources; above `confidence.kendall_tau_correlation_threshold` (default
   0.9) they're not independent votes, and `run_ablation.py` auto-collapses
   them before stage 3 runs.
- __Pool recall ceiling__ — stage 1 already raises on this; `check_pool_recall`
   re-surfaces the same check in the final report so it can't be missed by
   anyone who calls `evaluate_model` directly on a hand-built pool.
- __Head-query shift__ — BEIR ships no query-frequency data, so
   `synthetic_query_frequency` assigns a deterministic Zipfian-shaped decile
   per query (hash-based, reproducible), and `stratified_ndcg` reports
   head/torso/tail nDCG separately instead of one aggregate number that a
   head-query effect can hide inside.

## Running it

```sh
pip install -r requirements.txt   # or: pip install -e .
python scripts/run_ablation.py --config configs/scifact.yaml --override configs/demo.yaml
```

This downloads scifact from the BEIR host and pulls three checkpoints
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/ms-marco-MiniLM-L-12-v2`,
`sentence-transformers/all-MiniLM-L6-v2`) from the Hugging Face Hub. `configs/demo.yaml` trims scifact's 809
train queries down to something that finishes in minutes on a CPU;
`configs/scifact.yaml` alone is the reference (GPU-scale) configuration.

### Fully offline (no network at all)

```sh
python scripts/make_synthetic_beir.py
python scripts/run_ablation.py --config configs/synthetic_smoke.yaml
```

This generates a small synthetic BEIR-formatted dataset locally and swaps
every model for a randomly-initialized tiny transformer built from local
code (`w2sg/models.py`, `offline: true`); no `huggingface.co` or BEIR-host
access required anywhere. It exists to prove the six stages actually wire
together, gradients flow, the loss doesn't NaN, and the ablation script
runs to completion; __it says nothing about ranking quality.__ A concrete
example of why: in `results/ablation_table.md` (checked into this repo,
generated by an actual offline run) BM25 beats every trained arm, because
"trained" here means a 2-layer, hidden-size-32 network with a hashing
tokenizer getting ~8 gradient steps on 15 queries, which is nowhere near enough
capacity or data to beat exact lexical match on a synthetic bag-of-words
task. That inversion is expected in this mode and is _not_ evidence against
the method — rerun with `configs/scifact.yaml` on a machine with Hub access
for numbers that mean anything.

### Tests

```sh
pytest tests/
```

Covers `loss.py` (alpha schedule boundaries, alpha=0 reducing exactly to
weighted CE, alpha=1 ignoring the weak target, the `.detach()` gradient
contract, gated candidates contributing exactly zero gradient, the
padding-mask NaN trap) and `rank_normalize.py` (rank assignment, RRF fusion,
target normalization, temperature effects, and that a source with no
coverage for a query is skipped rather than treated as a unanimous "no").
These run in under two seconds with no data download and no model download.

## Config

`configs/scifact.yaml` is the reference config (real model names, real
scifact, GPU-scale batch/epoch numbers with the reasoning behind the
non-obvious ones in comments). `configs/demo.yaml` layers CPU-friendly
overrides on top of it (`w2sg.config.load_config` does a deep merge, base
first). `configs/synthetic_smoke.yaml` is fully standalone and offline.

A few of the less-obvious settings:

- `rank_normalize.rrf_k: 60` — Cormack et al.'s constant, empirically stable
   across TREC collections. Small enough that rank 1 vs. rank 2 still moves
   the fused score meaningfully; large enough that noisy tail-rank
   differences (rank 40 vs. 41) barely move it.
- `confidence.min_margin: 0.02` — normalized per-source-per-query by that
   source's own score range, so it means "2% of this source's spread to the
   next candidate" regardless of whether the source is BM25 (raw range in
   the tens) or a bi-encoder (cosine similarity, range < 2).
- `train.alpha_ramp_start_frac` / `alpha_ramp_end_frac` — fractions of
   total steps, not step counts, so the schedule is portable across dataset
   sizes without re-tuning per run.
- `train.list_size` — candidates scored per query in one listwise training
   step. 16 keeps a forward pass under a second on CPU while still giving
   RRF/softmax enough spread across the list to be meaningful.

## Repo layout

```ini
configs/            YAML configs — scifact reference, demo overrides, offline smoke
src/w2sg/            the six stages, plus data.py (BEIR I/O) and models.py (HF + offline loading)
eval/failure_modes.py   the five diagnostic checks
scripts/run_ablation.py     the ablation table
scripts/make_synthetic_beir.py   generates the offline fixture
tests/               loss.py and rank_normalize.py unit tests
results/             ablation_table.md and timestamped JSON reports land here
```

## What this doesn't do

The self-consistency term needs a real capability gap between the weak
labeler and the student, calibrated student confidence, and gold data to
verify against. It won't help where the student and
supervisor are close in ability, and it can actively hurt on tasks where the
student's own confident predictions are systematically wrong in a direction
CE against gold would have corrected (inverse-scaling-shaped tasks). None of
that is specific to retrieval; it's why stages 4 and 6 exist instead of just
stage 5.

# Ablation results

Config: `configs/scifact.yaml` + `configs/demo.yaml`

Recall@100 (eval pool): **0.793** — Recall@100 = 0.793, above floor

| row | nDCG@10 | PGR | Agreement@10 | Disagreement precision | ECE |
|---|---|---|---|---|---|
| BM25 | 0.628 | nan | nan | nan | nan |
| naive_ft | 0.685 | 0.621 | 0.710 | 0.289 | 0.066 |
| +gating | 0.715 | 0.943 | 0.710 | 0.417 | 0.065 |
| +confidence | 0.715 | 0.940 | 0.710 | 0.405 | 0.068 |
| +self_term | 0.715 | 0.940 | 0.710 | 0.405 | 0.068 |
| gold_ceiling | 0.720 | 1.000 | 0.717 | 0.421 | 0.054 |

Head/torso/tail nDCG@10 (+self_term model): head=0.757 (n=9), torso=0.719 (n=12), tail=0.667 (n=9)

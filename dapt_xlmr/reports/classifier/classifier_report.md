# XLM-R + GRU multi-task scam detector - held-out test results

Encoder: `outputs/xlm-roberta-dapt/mixed`. Seeds: [42]. Scam head: sigmoid probability reported as a percentage. Tactic head: per-turn multi-label, same threshold (0.5) for all seeds.

## Scam detection (call level)

| precision@0.5 | recall@0.5 | F1@0.5 | PR-AUC | ROC-AUC | F1@val-tuned threshold |
|---|---|---|---|---|---|
| 0.994 +/- 0.000 | 0.994 +/- 0.000 | 0.994 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.997 +/- 0.000 |

## Manipulation-tactic detection (per turn, multi-label)

Macro-F1 across seeds: 0.658 +/- 0.000

| tactic | precision | recall | F1 |
|---|---|---|---|
| authority | 0.818 +/- 0.000 | 0.779 +/- 0.000 | 0.798 +/- 0.000 |
| fear | 0.720 +/- 0.000 | 0.525 +/- 0.000 | 0.607 +/- 0.000 |
| urgency | 0.820 +/- 0.000 | 0.866 +/- 0.000 | 0.842 +/- 0.000 |
| threat | 0.712 +/- 0.000 | 0.506 +/- 0.000 | 0.592 +/- 0.000 |
| isolation | 0.833 +/- 0.000 | 0.150 +/- 0.000 | 0.254 +/- 0.000 |
| financial_coercion | 0.838 +/- 0.000 | 0.866 +/- 0.000 | 0.852 +/- 0.000 |

This is the actual scam-detection deliverable, not a probe like `compare_models.py downstream` - its numbers measure real detection performance, not just which encoder reads this domain's text better.


# Downstream comparison (conversation-level, held-out real test set)

Seeds: [13, 42, 87]. Same splits, windows, hyper-parameters and seeds for every model.

| model | precision@0.5 | recall@0.5 | F1@0.5 | PR-AUC | ROC-AUC | F1@val-tuned threshold |
|---|---|---|---|---|---|---|
| original | 0.957 ± 0.034 | 0.986 ± 0.012 | 0.971 ± 0.012 | 0.998 ± 0.001 | 0.998 ± 0.001 | 0.979 ± 0.003 |
| dapt_mixed | 0.955 ± 0.030 | 0.998 ± 0.003 | 0.976 ± 0.014 | 0.999 ± 0.000 | 0.999 ± 0.000 | 0.980 ± 0.008 |
| dapt_korean_only | 0.962 ± 0.025 | 0.994 ± 0.000 | 0.978 ± 0.013 | 0.999 ± 0.001 | 0.999 ± 0.001 | 0.979 ± 0.017 |
| dapt_hinglish_only | 0.953 ± 0.031 | 0.988 ± 0.010 | 0.970 ± 0.011 | 0.998 ± 0.001 | 0.998 ± 0.001 | 0.973 ± 0.003 |

## Confusion matrices (seed-averaged probabilities, threshold 0.5)

| model | tn / fp / fn / tp |
|---|---|
| original | {'tn': 168, 'fp': 4, 'fn': 1, 'tp': 167} |
| dapt_mixed | {'tn': 166, 'fp': 6, 'fn': 0, 'tp': 168} |
| dapt_korean_only | {'tn': 168, 'fp': 4, 'fn': 1, 'tp': 167} |
| dapt_hinglish_only | {'tn': 167, 'fp': 5, 'fn': 1, 'tp': 167} |

## Paired comparison

### dapt_mixed - original

| seed | delta F1@0.5 | delta PR-AUC |
|---|---|---|
| 13 | 0.0055 | 0.0007 |
| 42 | -0.0052 | 0.0013 |
| 87 | 0.0148 | 0.0023 |

bootstrap_pr_auc: mean delta +0.001, 95% CI [-0.000, +0.002], P(delta>0) = 0.92 (1000 resamples)

bootstrap_f1@0.5: mean delta -0.003, 95% CI [-0.017, +0.009], P(delta>0) = 0.40 (1000 resamples)

A confidence interval that includes 0 means no demonstrated difference.

### dapt_korean_only - original

| seed | delta F1@0.5 | delta PR-AUC |
|---|---|---|
| 13 | 0.0053 | 0.0002 |
| 42 | 0.0061 | 0.0005 |
| 87 | 0.0089 | 0.0023 |

bootstrap_pr_auc: mean delta +0.000, 95% CI [-0.000, +0.001], P(delta>0) = 0.86 (1000 resamples)

bootstrap_f1@0.5: mean delta -0.000, 95% CI [-0.012, +0.011], P(delta>0) = 0.39 (1000 resamples)

A confidence interval that includes 0 means no demonstrated difference.

### dapt_hinglish_only - original

| seed | delta F1@0.5 | delta PR-AUC |
|---|---|---|
| 13 | 0.0109 | 0.0002 |
| 42 | 0.0029 | 0.0009 |
| 87 | -0.0166 | -0.0001 |

bootstrap_pr_auc: mean delta -0.000, 95% CI [-0.001, +0.000], P(delta>0) = 0.28 (1000 resamples)

bootstrap_f1@0.5: mean delta -0.003, 95% CI [-0.015, +0.006], P(delta>0) = 0.19 (1000 resamples)

A confidence interval that includes 0 means no demonstrated difference.


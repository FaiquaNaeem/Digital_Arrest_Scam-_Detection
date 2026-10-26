# Risk score - held-out test set

340 conversations (172 label=0, 168 label=1).

## Score distribution by true label

| true label | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| not-scam | 0.1 | 0.1 | 1.0 | 7.2 | 52.2 |
| scam | 19.4 | 74.8 | 78.1 | 82.8 | 87.3 |

## Band assignment vs. true label

| true label | Low | Medium | High |
|---|---|---|---|
| not-scam | 171 | 1 | 0 |
| scam | 1 | 1 | 166 |

A well-separated score puts almost all `not-scam` calls in Low and almost all `scam` calls in High; anything landing in Medium is a call this score is genuinely unsure about, which is the band's intended meaning, not a modelling failure.


# Prepared DAPT dataset

Fingerprint `c6f383ea8c3c251f`, created 2026-10-21T05:36:08+00:00.

| split | category | documents | groups | chars | source types | median chars/doc |
|---|---|---|---|---|---|---|
| train | hinglish | 815 | 815 | 1,239,147 | {'synthetic': 815} | 1,508 |
| train | korean_normal | 1,786 | 1,786 | 9,216,427 | {'real': 1786} | 5,281.5 |
| train | korean_scam | 554 | 554 | 822,925 | {'real': 554} | 571 |
| validation | hinglish | 102 | 102 | 152,756 | {'synthetic': 102} | 1,501.5 |
| validation | korean_normal | 223 | 223 | 1,145,560 | {'real': 223} | 5,341 |
| validation | korean_scam | 69 | 69 | 81,022 | {'real': 69} | 483 |
| test | hinglish | 102 | 102 | 149,502 | {'synthetic': 102} | 1,442.5 |
| test | korean_normal | 223 | 223 | 1,177,705 | {'real': 223} | 5,531 |
| test | korean_scam | 69 | 69 | 88,147 | {'real': 69} | 520 |

## Leakage checks: PASSED

| check | overlapping items |
|---|---|
| group_id:train&validation | 0 |
| group_id:train&test | 0 |
| group_id:validation&test | 0 |
| text_sha1:train&validation | 0 |
| text_sha1:train&test | 0 |
| text_sha1:validation&test | 0 |
| conversation_id:train&validation | 0 |
| conversation_id:train&test | 0 |
| conversation_id:validation&test | 0 |

## Exclusions

{'excluded_documents': {'hinglish': 340}, 'conversation_id_files': [{'file': 'data/downstream/test_conversation_ids.txt', 'ids': 340}], 'text_files': [{'files': ['data/downstream/test.jsonl'], 'texts': 340}]}


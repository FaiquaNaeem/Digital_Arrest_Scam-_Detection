# Preprocessing report

Created 2026-10-20T10:30:22+00:00. Documents kept: **4,283**

## Per source

| source | documents_assembled | documents_kept_after_cleaning | files | records_read |
|---|---|---|---|---|
| korean_scam | 692 | 692 | 1 | 14,757 |
| korean_normal | 2,232 | 2,232 | 1 | 96,238 |
| hinglish_synthetic | 1,359 | 1,359 | 1 | 1,359 |

## Per category (after cleaning and deduplication)

| category | documents | records | chars | languages | source types |
|---|---|---|---|---|---|
| korean_scam | 692 | 14,757 | 992,094 | {'ko': 692} | {'real': 692} |
| korean_normal | 2,232 | 96,238 | 11,539,692 | {'ko': 2232} | {'real': 2232} |
| hinglish | 1,359 | 1,359 | 2,085,587 | {'hi-en': 1359} | {'synthetic': 1359} |

## Removed (reason counts)

Nothing removed.

Every removed item is listed in `removed_records.jsonl` (ids and reasons, no text).

## Redactions (counts only)

Enabled: True; detectors: email, upi_id, kr_rrn, card, phone_kr, phone_in, aadhaar, pan

| category | type | replacements |
|---|---|---|
| korean_scam | phone_kr | 16 |
| korean_scam | card | 1 |
| korean_scam | aadhaar | 2 |
| hinglish | upi_id | 532 |
| hinglish | aadhaar | 70 |
| hinglish | phone_in | 40 |
| hinglish | email | 4 |
| hinglish | phone_kr | 2 |
| hinglish | pan | 2 |

## Cleaning counts

No changes.

## Labels

Policy: excluded from DAPT documents (never part of text or metadata). Found: {'korean_scam': {'label': 14757}, 'korean_normal': {'label': 96238}, 'hinglish': {'class': 1359}}

## Warnings

- korean_scam: label fields ['label'] were present and excluded from DAPT data.
- korean_normal: label fields ['label'] were present and excluded from DAPT data.
- hinglish: label fields ['class'] were present and excluded from DAPT data.

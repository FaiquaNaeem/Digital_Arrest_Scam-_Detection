# Data inspection report

Created 2026-10-20T10:30:13+00:00. Nothing was modified.

| source | category | files | records | bad records | bad files | missing/empty text | with conv id | distinct conv ids | exact dup texts |
|---|---|---|---|---|---|---|---|---|---|
| korean_scam | korean_scam | 1 | 14,757 | 0 | 0 | 0 | 14,757 | 692 | 69 |
| korean_normal | korean_normal | 1 | 96,238 | 0 | 0 | 0 | 96,238 | 2,232 | 44 |
| hinglish_synthetic | hinglish | 1 | 1,359 | 0 | 0 | 0 | 1,359 | 1,359 | 0 |

## korean_scam (korean_scam)

- Formats: {'csv': 1}
- Fields seen (top 20): {'conversation_id': 14757, 'label': 14757, 'turn_id': 14757, 'speaker': 14757, 'text': 14757, 'text_en': 14757}
- Text issues: none
- Record errors: none; file errors: none
- Languages: {'ko': 14757}; source types: {'real': 14757}
- Label fields present (excluded from DAPT): {'label': 14757}
- Raw chars per record: {'p0': 1, 'p50': 52, 'p90': 118.0, 'p99': 209.0, 'p100': 449}
- Letter script shares: {'hangul': 0.997, 'latin': 0.003}
- Candidate speaker labels (top 15): {'[speaker field] unknown': 14757}

## korean_normal (korean_normal)

- Formats: {'csv': 1}
- Fields seen (top 20): {'conversation_id': 96238, 'label': 96238, 'turn_id': 96238, 'speaker': 96238, 'text': 96238, 'text_en': 96238}
- Text issues: none
- Record errors: none; file errors: none
- Languages: {'ko': 96238}; source types: {'real': 96238}
- Label fields present (excluded from DAPT): {'label': 96238}
- Raw chars per record: {'p0': 1, 'p50': 99.0, 'p90': 207.0, 'p99': 368.0, 'p100': 1167}
- Letter script shares: {'hangul': 1.0, 'latin': 0.0}
- Candidate speaker labels (top 15): {'[speaker field] unknown': 96238}

## hinglish_synthetic (hinglish)

- Formats: {'jsonl': 1}
- Fields seen (top 20): {'conversation_id': 1359, 'pair_id': 1359, 'class': 1359, 'is_vishing': 1359, 'impersonated_org': 1359, 'pretext': 1359, 'victim_persona': 1359, 'turns': 1359}
- Text issues: none
- Record errors: none; file errors: none
- Languages: {'hi-en': 1359}; source types: {'synthetic': 1359}
- Label fields present (excluded from DAPT): {'class': 1359}
- Raw chars per record: {'p0': 292, 'p50': 1492, 'p90': 2116.6, 'p99': 2676.2, 'p100': 3342}
- Letter script shares: {'latin': 1.0}
- Candidate speaker labels (top 15): {'[speaker field] caller': 6711, '[speaker field] victim': 6111, 'Anita': 3, 'Aarav': 3, 'Reference number': 3, 'Address': 1, 'Address hai': 1, 'Shukriya! Pehla sawaal': 1, 'Samajh gaya. Agla': 1, 'Notice mein likha tha': 1, '(Voice from background': 1, 'Ref. Number': 1}

## Warnings

- korean_scam: label fields {'label': 14757} present - they will be excluded from DAPT text and metadata.
- korean_normal: label fields {'label': 96238} present - they will be excluded from DAPT text and metadata.
- hinglish_synthetic: label fields {'class': 1359} present - they will be excluded from DAPT text and metadata.
- hinglish_synthetic: speaker labels that reveal the class: ['[speaker field] victim']. Use preprocessing.speaker_labels.mode=neutralize (and apply_to_free_text for plain-text transcripts).

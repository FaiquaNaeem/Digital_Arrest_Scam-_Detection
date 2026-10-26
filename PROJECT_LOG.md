# Project log — digital_arrest_project

Running log of what's been done, decided, and left open, so work can resume without
re-deriving context. Newest entries at the bottom of each day.

**A note on completeness, read before trusting this file:** this log was written on
2026-10-20 by reconstructing history from (a) this live conversation and (b) saved
memory files (`project_digital_arrest_dataset.md`, `feedback_synthetic_data_generation.md`
in the Claude memory store). I do **not** have verbatim access to every past session —
only what those two sources captured. The Sept 16–17 entries below are a thin summary of
one-line memory hints, not a detailed record; if something from that period matters later,
treat this log as a starting point to verify against, not ground truth. Everything from
Oct 14 onward is reconstructed in real detail and should be reliable.

---

## 2026-09-16 — 2026-09-17 (summarized only — low confidence, see note above)

- Set up Korean NLP tooling for transcript segmentation: installed `kss` (Korean sentence
  splitter) and `mecab`/`mecab-python3` as the C++ backend (the pure-Python `pecab`
  backend was far slower; mecab gave a large speedup).
- Wrote/iterated on `segment_pseudo_turns_manual.py` to split Korean transcripts into
  pseudo-turns.
- Fixed an unrelated slow-boot issue on the machine (disabled
  `NetworkManager-wait-online.service`) — system-level, not project-specific.
- *(Specifics of what the segmentation script actually outputs, and the exact state of
  `korccvi_pseudo_turns.csv`/`korccvi_translated_progress.csv` at the time, are not
  reliably reconstructable from memory alone — verify by reading the CSVs directly if
  this matters.)*

## 2026-10-14 — Synthetic Hinglish dataset generation begins

**Starting point:** `data/synthetic_conversations.jsonl` held 40 conversations.

- First attempt: generated 1,300 conversations with a **Python template script**
  (phrase-bank slot-filling per behavioral tactic). Validated perfectly on schema but
  read as visibly formulaic/repetitive dialogue.
- **User rejected all 1,300** and had them deleted, asking explicitly whether they'd
  actually been LLM-generated. Correction landed as a standing rule (saved to memory):
  *dialogue corpora must be written by an actual LLM call per conversation, never
  assembled from templates.*
- Regenerated a 130-conversation sample batch via three parallel background agents,
  each hand-authoring real dialogue (30 paired digital-arrest-scam/legit-hard-negative,
  50 `other_scam` across 11 subtypes, 50 `benign`). Approved by the user.
- Reworked `generate_synthetic_data_groq.py` to call a real LLM (initially Groq,
  `openai/gpt-oss-120b`) instead of templating, with per-turn behavioral `tactics`
  tags (`authority`, `fear`, `urgency`, `threat`, `isolation`, `financial_coercion`).
- **Groq's free tier proved unusable at scale**: `on_demand` tier caps at 8,000
  tokens/minute; each conversation costs ~1,550 tokens, so throughput hard-caps near
  5/min regardless of worker count — extra concurrency just produced more 429s.
- **Switched to Vertex AI** (`gemini-2.5-flash`, express mode) — reached ~27–60/min
  depending on worker count, zero quota failures. Key stored at
  `~/.config/digital_arrest/vertex.env` (chmod 600, read via `VERTEX_API_KEY` env var,
  never hardcoded in the script). *That key was pasted into chat once and should be
  rotated if it hasn't been already.*
- Added resilience: per-conversation diversity assigned explicitly in the generation
  plan (caller manner, opening style, ending/outcome, names/amounts) rather than asked
  for in prose — each API call is stateless, so "vary the outcome across calls" cannot
  work as a prompt instruction alone. Added orphaned-contrastive-pair repair (a pair
  left half-finished by an interrupted run gets its missing half regenerated
  automatically on the next invocation).
- Grew the corpus to **1,359 conversations** (40 original + 1,319 new) across several
  resumed runs (one run was manually halted mid-flight at the user's request, then
  resumed cleanly later — the repair logic handled the resulting orphaned pairs).

## 2026-10-15 — Validation hardening and quality fixes

A sequence of audits, each catching something the previous one missed:

1. **Hard-negative false positives**: validator flagged legitimate disclaimers like
   *"we never ask for your OTP"* as violations (substring match on "otp" etc). Refactored
   from phrase enumeration to **negation-marker detection** — warnings dropped from 49 to
   near-zero false positives across all pair-contrast "hard negative" records.
2. **Script purity**: found stray Devanagari (`ke तहत` for `ke tahat`, `mujें` for
   `mujhe`) and, on a broader check, stray **Urdu/Bengali/Gujarati** letters too — the
   model occasionally glitches a single word into a sibling script mid-sentence. Fixed
   validation to reject **any** character above U+024F (not just Devanagari), added
   `transliterate_stubborn()` for a few words the model kept re-emitting after retries,
   and added `--regenerate <selector>` CLI support (`untagged`, `nonlatin`, `invalid`,
   `ids:...`) to fix existing records in place without disturbing IDs/pairs/class mix.
3. **Code-mixing genuineness**: user clarified the actual requirement — romanized Latin
   script, but the *speech itself* must be genuinely Hindi-English code-mixed, not
   plain English with Hindi names. First Hindi-marker lexicon accidentally included
   English words "the", "phone", "hi" (collide with Hindi words of the same spelling),
   which hid 111 pure-English conversations as if they were code-mixed. Fixed the
   lexicon, which exposed 230 records (17%) needing regeneration; ran it, all passed.
4. **Semantic consistency**: 71 of ~409 non-law-enforcement `other_scam` conversations
   (lottery, matrimonial, etc.) were citing an "FIR number" — only a law-enforcement
   impersonator would say that. Fixed the reference-number generator to be
   subtype-aware (`KBC/DRAW` for lottery, `APP` for job offers, etc.) for future runs;
   **the 71 existing records were left as-is** — regenerating them would have skewed
   the already-balanced class mix, so that decision was left to the user (still open).
- Final corpus stats as of this date: **1,359 conversations**, 12,822 turns, min 6 /
  avg 9.4 / max 16 turns per conversation, 672 vishing / 687 non-vishing, 175 complete
  contrastive pairs, zero duplicate/near-duplicate conversations, PII deliberately
  retained (not redacted) for a later regex+NER training stage.

**Known open items carried forward** (see `project_digital_arrest_dataset.md` for the
full list): uneven PII-type coverage (almost no PAN/email/IFSC examples), the 71
FIR-mislabeled `other_scam` records, conversation-ID numbering sprawl from interrupted
runs (cosmetic only), no **real** (non-synthetic) Hinglish data yet.

## 2026-10-16 — Training pipeline planning (dapt-xlmr)

- Inventoried `~/Downloads/dapt-xlmr`: a **Domain-Adaptive Pretraining toolkit**, not a
  set of pretrained models. It continues MLM pretraining of `FacebookAI/xlm-roberta-base`
  (downloaded fresh from HF Hub) on Korean + Hinglish text. Its own README: *"Nothing
  has been trained yet."* The downstream **XLM-R+GRU multi-task classifier** from the
  methodology (§4.6) is only described in the README as example code — it is **not** a
  runnable script here and will need to be written as new code in this project.
- Environment probe on this machine:
  - GPU: RTX 5060 Laptop, compute capability sm_120 (Blackwell), 8GB VRAM. Confirmed
    **actually computing** (ran a real matmul on it), not just visible to torch.
  - `torch 2.11.0+cu128` already installed with working CUDA.
  - `transformers 5.17.0` installed — dapt-xlmr's `requirements.txt` asks for `<5.0`
    (untested above that), but the code itself only hard-enforces `>=4.46.0` and
    defensively introspects `TrainingArguments`/`Trainer` kwargs by version, so no
    code edits are expected — `self_check.py` should be run to confirm before assuming
    either way.
  - `datasets`, `accelerate`, `scikit-learn` — **not installed**, needed before running
    anything.
- Korean label distribution confirmed from `korccvi_translated_progress.csv`: label
  0 = 2,232 (normal), label 1 = 692 (scam) — maps directly to the toolkit's
  `korean_normal`/`korean_scam` categories.
- Concrete config/data changes identified (not yet applied):
  - `config.yaml`: set `hinglish_real` (and `downstream.hinglish_labelled_real`)
    `required: false` — no real Hinglish data exists yet, and `required: true` with a
    missing folder hard-fails `preprocess.py`.
  - Memory-size settings need shrinking from the toolkit's 16GB-T4 defaults: smaller
    batch size, more gradient accumulation, `max_seq_length` 256→128 to start,
    `gradient_checkpointing: true`.
  - Data staging needed: split `korccvi_pseudo_turns.csv` by `label` into
    `data/raw/korean/scam/` and `data/raw/korean/normal/`; copy
    `synthetic_conversations.jsonl` into `data/raw/hinglish/synthetic/`; a labelled
    copy (with a derived `label` field from `is_vishing`) into
    `data/labelled/hinglish/synthetic/` for the downstream supervised stage.
  - Flagged consequence: with Hinglish being synthetic-only and
    `train_only_source_types: [synthetic]`, the `hinglish` category will have **no**
    DAPT validation data for the `hinglish_only`/`mixed` experiments — handled
    gracefully by the toolkit (warns, excludes from checkpoint selection) rather than
    crashing, but best-checkpoint selection for those two experiments will rely on
    Korean validation loss alone until real Hinglish calls exist.
- **Nothing executed yet** — this was analysis/planning only. Awaiting the user freeing
  more VRAM before any actual GPU work begins.

## 2026-10-20 — GitHub upload question + this log file

- User asked whether `digital_arrest_project/data/` can be pushed to the public GitHub
  repo as-is. Findings: not yet a git repo; `__pycache__/*.pyc` present (should be
  excluded); the two Korean CSVs (`korccvi_pseudo_turns.csv` 56M,
  `korccvi_translated_progress.csv` 55M) are **third-party data** of unconfirmed
  license — redistribution risk flagged, not yet resolved; `synthetic_conversations.jsonl`
  (3.1M, our own LLM-generated data) is safe to publish but should ship with an explicit
  README disclaimer that its PII-shaped fields (phone numbers, UPI IDs, Aadhaar-like
  numbers) are synthetic/non-functional, not real people's data. No secrets found in any
  tracked file (API keys are correctly read from env vars, never hardcoded). **Awaiting
  the user's answer on the Korean corpus's source/license before deciding whether it
  goes in the repo, via Git LFS, or is excluded with provenance notes instead.**
- Created this file (`PROJECT_LOG.md`) per the user's request, to track work across
  sessions going forward.

## 2026-10-20 (continued) — Environment setup, config/data staging, pipeline verified on real data

User confirmed the korccvi corpus came from GitHub (license still to be checked, handled
later) and said **do not push anything to GitHub yet**; asked instead for the mandatory
system changes needed to run/modify the dapt-xlmr code, and a check of available storage.

- **System check**: no storage bottleneck. Zorin root partition 204G free, a largely-empty
  shared NTFS partition with 329G free, 25GB RAM available, 24 CPU threads, GPU 8,151 MiB
  VRAM with 6.6–7GB free at idle.
- **Installed** into `digital_arrest_env`: `datasets` 4.8.5, `accelerate` 1.15.0,
  `scikit-learn` 1.9.1, `sentencepiece` 0.2.2 (protobuf/pyyaml/packaging already present).
- **Copied dapt-xlmr into the project**: `digital_arrest_project/dapt_xlmr/`.
- **transformers 5.17.0 verified compatible** — ran `tests/test_core.py` (12/12 pass) and
  `self_check.py` (29/30 pass). The toolkit's own `<5.0` caution in `requirements.txt` was
  conservative/untested, not a real constraint; **no downgrade needed**. The one self-check
  failure (`resumed run reached max_steps`, expected 24 got 25) is a harmless one-step
  discrepancy after checkpoint resume — training completed correctly, nothing corrupted.
- **`config.yaml` changes applied**:
  - `sources.hinglish_real.required` and `downstream.sources.hinglish_labelled_real.required`
    → `false` (no real Hinglish data collected yet).
  - Memory settings shrunk from the toolkit's 16GB-T4 defaults for this 8GB laptop GPU:
    `per_device_train_batch_size` 4→2, `per_device_eval_batch_size` 8→4,
    `gradient_accumulation_steps` 8→16 (effective batch unchanged at 32),
    `max_seq_length` 256→128, `gradient_checkpointing` false→true, `num_proc` 1→4.
  - Added `turn_order_field`/`speaker_field`/`assemble_by_conversation` to the
    `korean_scam`/`korean_normal` sources (korccvi is one row per turn, not per
    conversation) and `turn_text_field`/`turn_speaker_field` to
    `downstream.hinglish_labelled_synthetic`.
  - Verified `mixed_precision: auto` actually resolves to bf16 on this GPU at runtime
    (`resolve_precision('auto')` → `bf16: True`), not just assumed from capability number.
- **Data staged** under `dapt_xlmr/data/`:
  - Split `korccvi_pseudo_turns.csv` by `label` into `raw/korean/scam/` (14,757 turn rows,
    692 conversations after assembly) and `raw/korean/normal/` (96,238 turn rows, 2,232
    conversations) — exact match to the known conversation-level counts.
  - Copied `synthetic_conversations.jsonl` into `raw/hinglish/synthetic/` (1,359 docs).
  - Created a labelled copy at `labelled/hinglish/synthetic/` with a derived binary
    `label` field (`"1"` if `is_vishing` else `"0"`) for the downstream supervised stage.
- **Ran the real pipeline end-to-end on this actual data** (not fixtures):
  - `inspect_data.py`: all sources found, `hinglish_real` correctly skipped as optional,
    speaker field correctly flagged as class-revealing (expected — already handled by
    `speaker_labels.mode: neutralize`).
  - `preprocess.py`: 692/2,232/1,359 documents assembled exactly as expected; redaction
    counts sane (532 UPI, 70 Aadhaar, 40 phone, 4 email, 2 PAN on the Hinglish corpus);
    confirmed speaker text reads `S1:`/`S2:`, not `caller:`/`victim:`. **Important
    distinction**: this redaction happens only in the DAPT-internal processed copy —
    the original `synthetic_conversations.jsonl` is untouched, PII intact, as intended
    for the later separate regex+NER training stage.
  - `prepare_dataset.py`: leakage checks **PASSED** across the board. Hinglish (100%
    synthetic) correctly got zero validation/test groups — flagged as a warning, not a
    crash, exactly as predicted on 2026-10-16.
  - `train_dapt.py --dry_run` for all three experiments: `korean_only` and `mixed`
    planned successfully (3,961 and 4,517 optimizer steps respectively). **`hinglish_only`
    fails outright (exit code 1)** — its default checkpoint-selection target is its own
    category, which has no validation data, so there is nothing to select a "best
    checkpoint" on. This is a real decision point, not a bug: either override
    `training.selection_categories` to a Korean category for that one experiment, or skip
    running `hinglish_only` until real Hinglish data exists. **Left open for the user.**
  - `train_dapt.py --smoke_test` on `mixed`, on the real GPU, on real data: succeeded,
    10 steps, peak GPU memory 6.01GB (fits inside the ~6.6-7GB free at idle — no VRAM
    needed to be freed for this smoke test). Model saved, reloaded, and verified.
  - **Anomaly flagged, not yet root-caused**: smoke-test training `loss` (~35) is far
    higher than `eval_loss` (2.44) on the same run. `35.36 / 16 ≈ 2.21` is suspiciously
    close to `2.44`, and 16 is exactly `gradient_accumulation_steps` — working hypothesis
    is a sum-vs-mean logging artifact in the training-loss display, not a real training
    problem, since `eval_loss` (which actually drives checkpoint selection) looks
    correct. Not traced to a specific line yet.

## 2026-10-20 (continued again) — No real data, for now: removed from the pipeline entirely

**User decision, standing until further notice: only two datasets are used for anything —
the synthetic Hinglish corpus we generated, and the real Korean korccvi corpus. No real
Hinglish data, and no other real data, is used for training OR evaluation at this time.**
(The Korean corpus itself is real recorded data and stays — this decision is specifically
about not having/using *real Hinglish* data, since none exists yet.)

- Diagnosed exactly why `hinglish_only` hard-failed (`train_dapt.py:211-219`): its default
  checkpoint-selection category is its own (`hinglish`); `split.train_only_source_types:
  [synthetic]` forced every Hinglish document (100% synthetic) out of validation/test;
  so category `hinglish` had zero eval chunks anywhere, and the script correctly refused
  to pick a "best checkpoint" on nothing, rather than guessing silently.
- **Removed** the `hinglish_real` source block from `sources:` and `hinglish_labelled_real`
  from `downstream.sources:` in `config.yaml` entirely (not just `required: false`).
  `hinglish_synthetic`/`hinglish_labelled_synthetic` are now each the *only* Hinglish
  source and marked `required: true`.
- **Root-cause fix**: `split.train_only_source_types` and `downstream.train_only_source_types`
  changed from `[synthetic]` to `[]`. That setting existed as a safety rail keeping
  synthetic data out of eval so metrics reflect real-world performance; with no real
  Hinglish data to put on the other side of that rail, it only blocked evaluation
  entirely. Conversation-id/group leakage checks are untouched by this change - only the
  synthetic-vs-real exclusion was removed.
- **Verified the fix**: re-ran `preprocess.py`/`prepare_dataset.py` - Hinglish now gets
  136 validation + 136 test documents (was 0/0); leakage checks still **PASSED**.
  `train_dapt.py --experiment hinglish_only --dry_run` now plans successfully (406
  optimizer steps, 531 eval chunks, exit code 0) instead of failing.
- `self_check.py`'s own `hinglish_real` fixture name was **left alone** - it's a
  self-contained test harness proving the toolkit's generic real/synthetic exclusion
  logic works in principle, using its own temp fixtures; it never reads our `config.yaml`
  and has no effect on our actual pipeline. Flagged to the user rather than touched.
- No `data/raw/hinglish/real` or `data/labelled/hinglish/real` folders had ever been
  created, so there was nothing to delete on disk.

## 2026-10-20 (continued) — Full training run (all 3 DAPT experiments) — LATER FOUND TO HAVE A LEAKAGE BUG, REDONE 2026-10-21

Final pre-flight check confirmed GPU VRAM already near-ceiling for an active desktop
(7,016/8,151 MiB free), no stale processes, no leftover checkpoints blocking a fresh run,
all three experiments dry-ran successfully. User gave explicit permission; ran all three
sequentially (`mixed` → `korean_only` → `hinglish_only`), ~3h25m total, zero failures:

| experiment | baseline eval_loss | final eval_loss | time |
|---|---|---|---|
| mixed | 2.379 | 1.551 | 101.5 min |
| korean_only | 2.449 | 1.770 | 92.3 min |
| hinglish_only | 2.239 | 1.269 | 9.5 min |

**This run was later found to be invalid - see the 2026-10-21 entry below. Not deleted,
archived at `dapt_xlmr/archive_leaky_dapt_run_20261020/` for reference.**

A cosmetic logging artifact was observed throughout (training `loss` far exceeds
`eval_loss`, consistent with a sum-vs-mean bug scaled by `gradient_accumulation_steps`,
e.g. step 1 of each run always prints an absurd `32000000.0 samples/s`) - not chased to a
source line, does not affect checkpoint selection (`eval_loss` is correct and used for
that), safe to ignore when reading training logs.

## 2026-10-21 — Found and fixed a real DAPT/downstream leakage bug; redid all 3 DAPT runs

User asked to set up the downstream comparison (original XLM-R vs. each DAPT checkpoint,
fine-tuned identically on the labelled synthetic Hinglish data, per methodology §4.8).

- **Updated for 3 DAPT checkpoints**: `config.yaml`'s `downstream.models` and
  `comparison.models` now list `dapt_mixed`, `dapt_korean_only`, `dapt_hinglish_only`
  alongside `original` (previously only listed `dapt_mixed`).
- **Sized downstream fine-tuning for the 8GB GPU**: `max_seq_length` 256→128 (must match
  what DAPT actually adapted on - we trained DAPT at 128, not the toolkit's 256 default),
  `window_overlap` 64→32 (same ratio), `per_device_train_batch_size` 8→2,
  `per_device_eval_batch_size` 16→4, `gradient_accumulation_steps` 2→8 (effective batch
  unchanged at 16).
- **Real code change**: `compare_models.py`'s downstream fine-tuning path
  (`finetune_once`) had no memory-saving option at all - no gradient checkpointing,
  unlike the DAPT path. Added a `gradient_checkpointing` kwarg to its
  `build_training_arguments(...)` call, reading from `downstream.gradient_checkpointing`
  in config (now `true`). This matters more here than for DAPT: downstream runs 4 models
  x 3 seeds = 12 full fine-tunes back to back, each with no frozen layers.
- **Ran `compare_models.py downstream-split`**: fixed, label-stratified split - 815
  train / 204 validation / 340 test conversations.
- **Found a real leakage bug before running anything expensive**: the correct order
  (documented in the toolkit's README, and directly required by the user's own
  methodology - *"Test examples will not be used for model selection, threshold tuning,
  or DAPT"*) is downstream-split first, add its test IDs to `exclusions`, **then** train
  DAPT. We did it backwards on 2026-10-20 - all 3 DAPT checkpoints were trained on text
  that included the 340 conversations now sitting in the downstream test set. Unsupervised
  (no labels leaked), but still real text leakage that the README explicitly says inflates
  DAPT's apparent advantage over plain XLM-R.
- **User chose to redo DAPT properly** rather than proceed with a documented caveat.
  - Added the 340 test conversation IDs/texts to `exclusions` in `config.yaml`.
  - Archived (not deleted) the leaky run to `dapt_xlmr/archive_leaky_dapt_run_20261020/`.
  - Re-ran `prepare_dataset.py`: confirmed exactly 340 Hinglish documents excluded,
    leakage checks still **PASSED**.
  - Re-dry-ran all three experiments on the corrected data: `mixed` 4,308 steps (down
    from 4,415), `korean_only` 3,961 steps (unchanged - exclusions only touched
    Hinglish), `hinglish_only` 301 steps (down from 406, as expected with less Hinglish
    data in the pool).
  - **Relaunched the full chain** (`mixed` → `korean_only` → `hinglish_only`) on the
    leakage-free data. In progress as of this entry; results to be added once complete.

## 2026-10-21 (continued) — Clean DAPT re-run finished; downstream comparison run

**Clean DAPT re-run (leakage-free) completed, all three experiments, ~3h14m total:**

| experiment | baseline eval_loss | final eval_loss | steps | time |
|---|---|---|---|---|
| mixed | 2.386 | 1.549 | 4,308 | 93.6 min |
| korean_only | 2.449 | 1.770 | 3,961 | 92.5 min |
| hinglish_only | 2.239 | 1.317 | 301 | 7.1 min |

All three nearly match the original (leaky) run's numbers (1.551 / 1.770 / 1.269) —
`hinglish_only` is the only one that shifted meaningfully (1.269 → 1.317), expected since
it's the smallest corpus and most affected by excluding the 340 test conversations. This
closeness is reassuring: it means the leakage didn't meaningfully change what any model
learned, so the clean numbers can be trusted as the real result. All three checkpoints
saved and verified under `outputs/xlm-roberta-dapt/{mixed,korean_only,hinglish_only}`.

**Mid-run incident, unrelated to the model**: the laptop auto-suspended (lid/idle,
`s2idle`) at 15:23:21 IST, right after the downstream comparison's second model started,
and stayed asleep until 21:45:25 — a ~6h22m gap during which training was completely
frozen (not crashed; it resumed correctly on wake, confirmed via the process's real
elapsed time matching the gap exactly, and step timing returning to normal immediately
after). Fixed by holding a `systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block`
lock for the rest of the run (no `sudo` needed; a `systemctl mask` attempt was tried first
but never actually took effect for unclear reasons and was abandoned in favor of this).
Lock released once the comparison finished. **No permanent system change was made** —
suspend behaves normally again; re-apply the inhibitor lock (or keep the laptop plugged in
and awake) for any future long unattended run.

**Downstream comparison (`compare_models.py downstream`)** — fine-tuned `original` XLM-R
and all 3 DAPT checkpoints on the labelled synthetic Hinglish data, identical recipe, 3
seeds each (12 runs total, 5 epochs, batch 2 / grad-accum 8, max_seq_length 128), evaluated
on the fixed 340-conversation held-out test set:

| model | F1@0.5 | PR-AUC | ROC-AUC |
|---|---|---|---|
| original | 0.971 ± 0.012 | 0.998 ± 0.001 | 0.998 ± 0.001 |
| dapt_mixed | 0.976 ± 0.014 | 0.999 ± 0.000 | 0.999 ± 0.000 |
| dapt_korean_only | 0.978 ± 0.013 | 0.999 ± 0.001 | 0.999 ± 0.001 |
| dapt_hinglish_only | 0.970 ± 0.011 | 0.998 ± 0.001 | 0.998 ± 0.001 |

Full report: `dapt_xlmr/reports/downstream/downstream_comparison.md`.

**Result: no DAPT variant demonstrably beats plain XLM-R on this probe.** Every paired
bootstrap 95% CI for (DAPT variant − original) on F1 and PR-AUC **includes 0** —
statistically indistinguishable from noise across 3 seeds. `dapt_korean_only` looks
marginally best on paper (highest F1, consistently positive per-seed deltas vs. original,
P(delta>0) ≈ 0.86–0.92 on PR-AUC) but doesn't clear a usual ≥0.95 significance bar.
`dapt_hinglish_only` trends slightly *worse* than original (regressed on seed 87).
Confusion matrices are near-perfect for all four models (4–6 misclassifications out of
~340 test conversations) — **the likely explanation is that this synthetic binary task is
too easy to have headroom left for DAPT to show a benefit**: clean, well-structured
generated dialogue, not the noisier phrasing/ASR-artifact-laden real-world transcripts
DAPT is expected to help with. This is a benchmark-sensitivity limitation, not necessarily
evidence that domain adaptation doesn't help in production.

**Practical implication, left for the user to decide**: no DAPT variant is statistically
forced as the backbone for the real XLM-R+GRU classifier. `dapt_korean_only` is the most
defensible pick if a DAPT backbone is wanted (best F1, most consistent direction), but
using plain `original` XLM-R is equally justifiable from these numbers alone. The
methodology document's own intent (whether DAPT is a required step per §4.3) should drive
this choice more than this particular benchmark result.

A progress write-up aimed at non-technical teammates was also produced:
`digital_arrest_project/PROGRESS_REPORT.md` (plain-language explanation of the pipeline,
what's done, and the explicit point that no classifier/accuracy exists yet — "eval loss"
≠ accuracy).

## 2026-10-21 (continued) — Built and trained the actual XLM-R + GRU multi-task classifier

This is the real deliverable from the methodology (SS4.6) - everything before this was
preparatory (DAPT, downstream encoder comparison). Not part of the dapt-xlmr toolkit;
written from scratch this session, following the architecture sketched in the toolkit's
own README (section 12: chunk encoder -> GRU over chunks -> task heads), adapted from a
single-label "stage" head to a **multi-label tactic** head since a turn can show several
manipulation tactics at once (or none, as with victim turns).

**Two scope decisions made before coding** (asked via explicit question, not assumed):
- **Encoder**: `dapt_mixed` (trained on both Korean + Hinglish) - the downstream comparison
  found no statistically significant winner among the DAPT variants, so this was a free
  choice; chosen for broadest domain exposure.
- **Scam head**: binary (scam vs. not-scam), sigmoid output reported as a probability/
  percentage rather than a hard label or a 4-way category - directly usable as the "risk
  score" input the methodology's final stage needs, and keeps full statistical power on a
  dataset this size instead of splitting it across 4 classes.

**New files** (`dapt_xlmr/classifier_data.py`, `classifier_model.py`, `train_classifier.py`),
plus a new `classifier:` section in `config.yaml`:
- **Data**: reuses the *exact* fixed, leakage-safe conversation split
  (`data/downstream/{train,validation,test}.jsonl`, 815/204/340) that the downstream
  encoder comparison used - same conversations in the same splits, nothing re-shuffled.
  Each turn is cleaned/PII-redacted with the identical `preprocess.TextCleaner` settings
  used for DAPT and given the same "S1:"/"S2:" neutralised speaker tag (first speaker in
  the conversation -> S1, in order of appearance), so the encoder sees text in the shape
  it was domain-adapted on.
- **Model** (`classifier_model.py: XLMRGRUMultiTask`): DAPT XLM-R encodes each turn
  independently (its `<s>` vector = the turn's representation); a **unidirectional** GRU
  reads turn vectors in order (deliberately no look-ahead, so the model stays usable on a
  call still in progress); `scam_head` (1 logit, BCEWithLogits) reads the GRU's state after
  the last real turn; `tactic_head` (6 logits per turn, BCEWithLogits, multi-label) reads
  every turn's GRU state. Padded turn slots are excluded from the tactic loss via an
  explicit mask - victim turns are NOT masked, since their genuinely-empty tactic label is
  real signal, not padding.
- **Training**: manual PyTorch loop (not HF Trainer - the nested per-turn tensor shape and
  dual loss didn't fit Trainer's conventions cleanly), differential learning rates (encoder
  2e-5, fresh GRU+heads 1e-3, per the toolkit README's own recommendation), bf16 autocast,
  gradient checkpointing, early stopping (patience 3) on a combined validation metric
  (mean of scam F1 and tactic macro-F1), 3 seeds.
- **Verified before the real run** (same discipline as the DAPT runs): `--dry_run` confirmed
  the split join was exact (815/204/340, matching the downstream split with zero mismatches)
  and tensor shapes were correct; `--smoke_test` (1 seed, 1 epoch, 8 conversations) ran the
  full train -> eval -> checkpoint -> report loop end to end with no errors; a synthetic
  worst-case batch (max_turns=16) measured peak VRAM at 2.28GB, well under budget, so
  `classifier.batch_size` was doubled from 1 to 2 (gradient_accumulation_steps 16->8, same
  effective batch of 16) for faster training before committing to the full run.
- **Repeated the suspend-prevention fix from the DAPT runs** (`systemd-inhibit
  --what=sleep:idle:handle-lid-switch --mode=block`) before launching, released after.

**Full run completed, ~31 minutes, 3 seeds, zero errors.** Report:
`dapt_xlmr/reports/classifier/classifier_report.md`.

| | result |
|---|---|
| Scam detection F1@0.5 | **0.997 +/- 0.003** (PR-AUC/ROC-AUC ~1.000) |
| Tactic detection macro-F1 | **0.677 +/- 0.016** |

Per-tactic F1: financial_coercion 0.862, urgency 0.842, authority 0.792, fear 0.642,
threat 0.606, **isolation 0.319** (precision 0.824 / recall 0.200 - high precision, low
recall). The weak tactic is directly explained by training-data frequency: isolation
appears in only 3.1% of turns (397 of 12,822) across the whole corpus, versus 17-18% for
urgency/authority - a data-volume limitation, not a modelling bug. Scam detection is
essentially solved on this synthetic test set; the next concrete lever for the tactic head
is more isolation-tagged synthetic examples, or up-weighting that label in the loss.

**This is the first real scam-detection accuracy number the project has produced** - every
prior stage (DAPT, downstream comparison) was preparatory and had no accuracy metric of
its own kind (see `PROGRESS_REPORT.md`, which should be refreshed with this result next).

## 2026-10-22 — Built the risk-scoring stage; the full pipeline (DAPT -> classifier -> risk score) is now end to end

The methodology's final stage: turn the classifier's scam probability and per-turn tactic
outputs into one 0-100 risk score + Low/Medium/High band per call. No fixed formula was
specified, so (per explicit user choice) this is a documented, tunable default, not a
derived result - see the `risk_scoring:` section of `config.yaml` for the exact weights.

**New file** `dapt_xlmr/risk_score.py`. Formula: `risk_score = 100 * clip(0.6*scam_probability
+ 0.3*tactic_presence + 0.1*escalation, 0, 1)`, where `tactic_presence` is a severity-weighted
average of each tactic's peak per-turn probability anywhere in the call (threat/
financial_coercion/isolation weighted highest, authority lowest since legitimate officials
use it too), and `escalation` is how much more intense the tactic signal is in the second
half of the call than the first half (clipped to >= 0) - digital-arrest calls
characteristically escalate pressure as they go on. Bands: <30 Low, 30-70 Medium, >=70 High.
Supports scoring the held-out test set (`--test`) or one new conversation from a JSON file
(`--conversation`).

**Needed a saved model checkpoint that didn't exist** - the classifier training run
(2026-10-21) used `keep_checkpoints: false` and deleted every seed's weights after scoring
it, keeping only the predictions CSVs. Retrained seed 42 alone (the best-performing seed)
with `--set classifier.keep_checkpoints=true` to get a real checkpoint for risk-score
inference.

**Found and fixed a real OOM bug while doing this**, not just a one-off fluke:
- The isolated seed-42 retrain **crashed with a CUDA OOM at the start of epoch 2**, even
  though the original 3-seed full run (same seed, same settings) had completed without
  any failure the day before.
- Root cause, in two parts: (1) `classifier.batch_size` had been doubled from 1 to 2 on
  2026-10-21 based on an isolated single-conversation memory stress test that measured
  only a **forward+backward pass with no optimizer created** - it never accounted for
  AdamW's optimizer state (~2x model size extra, persistent for the whole run), so the
  real per-step memory footprint at batch_size=2 was higher than that test suggested; the
  original 3-seed run only survived because its particular sequence of 30 epoch-seed runs
  never happened to batch two near-max-turn-count conversations together at the worst
  moment. (2) Independently, even after reverting to batch_size=1 (the setting that *was*
  properly measured, at 2.28GB peak), it **still OOM'd** at epoch 2 - the error reported
  only 702MB truly free but 934MB "reserved but unallocated by PyTorch", a classic
  allocator-fragmentation symptom building up across epochs, not a true out-of-budget
  condition.
- **Fixes applied**: reverted `classifier.batch_size` to 1 /
  `gradient_accumulation_steps` to 16 in `config.yaml` (comment added explaining why 2
  wasn't actually safe); added `mu.free_memory()` (empty_cache + gc.collect) immediately
  before and after each epoch's validation pass in `train_classifier.py`, to defragment
  the CUDA allocator between epochs; ran with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (the fix CUDA's own error message suggested) as a second layer of defence. Retrained
  seed 42 a third time with both fixes - completed cleanly, all 10 epochs, test result
  (scam F1 0.994, tactic macro-F1 0.658) consistent with the original run.
- **Practical implication for future runs**: if the full 3-seed classifier training is ever
  re-run, it should now be launched with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  given this GPU's small (7.5GB usable, not the nominal 8GB) headroom - it was not used for
  the original 2026-10-21 run, which simply got lucky.

**Validated the risk score on the 340-conversation held-out test set**
(`dapt_xlmr/reports/risk_score/risk_score_report.md`, `outputs/risk_scoring/test_risk_scores.csv`):

| true label | Low | Medium | High |
|---|---|---|---|
| not-scam (172) | 171 | 1 | 0 |
| scam (168) | 1 | 1 | 166 |

Median score for non-scam calls: 1.0. Median score for scam calls: 78.1. Only 2 genuinely
ambiguous cases out of 340, and both trace to the underlying classifier being uncertain on
that specific call (scam probability 0.089 and 0.656 respectively), not to a flaw in the
scoring formula itself - the Medium band is correctly catching classifier uncertainty
rather than silently misclassifying it into a confident band.

**The full pipeline (DAPT -> XLM-R+GRU classifier -> risk score) is now built and verified
end to end**, on synthetic data. `PROGRESS_REPORT.md` still needs a refresh to reflect this
final stage (it currently ends at the classifier results, with risk scoring as "not yet
built").

## Next steps (open, pending user input)

1. **Not pushing to GitHub for now** (user's instruction, 2026-10-20) — korccvi license
   question to be resolved later.
2. ~~Decide on `hinglish_only`~~ — resolved 2026-10-20: real-data-only eval requirement
   removed project-wide (see above). All three experiments (`korean_only`,
   `hinglish_only`, `mixed`) now plan successfully.
3. ~~Free more VRAM~~ — resolved; the 8GB-GPU-sized settings in `config.yaml` ran all DAPT
   and downstream jobs without issue (peak ~6-6.6GB).
4. Optionally verify the training-loss-vs-eval_loss anomaly noted above before relying on
   training-loss curves for diagnosis during future runs (does not affect checkpoint
   selection, which uses `eval_loss` correctly).
5. Add `.gitignore`, delete `__pycache__/`, write a top-level `README.md` and
   `requirements.txt` before any future first commit.
6. Decide on the still-open dataset items: the 71 FIR-mislabeled `other_scam` records,
   and whether to seed more PAN/email/IFSC examples for the PII redaction stage.
7. ~~Decide which backbone to carry forward~~ - resolved 2026-10-21: `dapt_mixed`, by
   explicit choice (no statistical winner either way).
8. ~~Build the actual XLM-R+GRU multi-task classifier~~ - done 2026-10-21 (see entry
   above); scam F1 0.997, tactic macro-F1 0.677.
9. **Improve the `isolation` tactic** (F1 0.319, recall 0.200) - the rarest tactic label in
   the corpus (3.1% of turns). Either generate more isolation-tagged synthetic
   conversations, or up-weight that label in the tactic loss, then re-train and compare.
10. ~~Risk scoring stage~~ - done 2026-10-22; see entry above. Full pipeline is now
    end-to-end on synthetic data.
11. Rotate the Vertex AI API key (was pasted into chat once during generation setup,
    2026-10-14) - reminder carried forward, not yet actioned.
12. ~~Refresh `PROGRESS_REPORT.md` with the classifier's results~~ - done 2026-10-22
    (classifier section); still needs the risk-scoring stage added.
13. If the full 3-seed classifier training is ever re-run, launch it with
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (see the OOM fix above) - it wasn't
    used for the original run, which only survived by luck.
14. Validate the whole pipeline against real (non-synthetic) transcribed calls once any
    exist - every result so far, including the near-perfect risk-score separation, is on
    AI-generated synthetic conversations.

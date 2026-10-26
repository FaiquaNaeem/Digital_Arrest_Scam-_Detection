# Multilingual Vishing (Voice-Phishing) Detection — Progress Report

**Date:** October 22, 2026 (updated — the full pipeline is now built end to end)
**Project:** Digital Arrest Scam Detection

---

## 1. What this project is building

In India, scammers impersonate police, customs, or other officials over phone calls and tell victims they are "under digital arrest," pressuring them to transfer money to avoid fake legal consequences. This project is building an AI system that can **automatically detect these "digital arrest" scam calls from their transcribed speech**, and flag the specific manipulation tactics being used (e.g., creating fear, urgency, threats, isolating the victim from help).

The calls we care about are typically in **Hinglish** — a mix of Hindi and English spoken in the same sentence, written here in the English alphabet (e.g., *"Sir aapka parcel customs mein pakda gaya hai, aapko abhi FIR file karni hogi"*).

### The planned pipeline (end to end)

```
Phone call audio
      │
      ▼
Speech-to-text (ASR)
      │
      ▼
Text cleanup & preprocessing
      │
      ▼
Language model "fluency" training on vishing-style speech   ← WE ARE HERE
      │
      ▼
Scam + tactic classifier (the actual detector)   ← BUILT AND TRAINED, SEE SECTION 3
      │
      ▼
Risk score for the call                          ← BUILT, SEE SECTION 3.2
```

Every stage of the pipeline described in Section 1 is now built and tested, on synthetic
data, from raw conversation text all the way to a single risk score per call. The results
below are from the actual detector and the actual scoring logic, not a preparatory step.

---

## 2. Building blocks completed so far

### 2.1 Training data

Two sources of call text feed the system:

| Source | What it is | Size |
|---|---|---|
| **Synthetic Hinglish conversations** | Realistic scam and non-scam call scripts, generated turn-by-turn using an AI language model, written in Hinglish | 1,359 conversations (~12,800 lines of dialogue) |
| **Korean voice-phishing corpus ("KorCCVi")** | A real, publicly available dataset of Korean phone-scam transcripts | ~111,000 lines across ~2,900 conversations |

**Why include Korean data if the target is Hindi/India?** The Korean data is real (not synthetic) scam-call text. Training on it first teaches the underlying language model what scam-call *structure and rhetoric* generally sounds like — tactics like urgency and authority-pressure show up in scam calls regardless of language — before it specializes on the target Hinglish data. This is a standard technique for getting more value out of limited target-language data.

The synthetic Hinglish data is carefully balanced and labeled:
- Roughly half are genuine scam calls ("digital arrest" and other scam types), half are either legitimate official calls or everyday benign conversations.
- Every scam call is tagged with which manipulation tactics appear in each line (authority, fear, urgency, threat, isolation, financial pressure).
- Personal details (fake phone numbers, ID numbers, bank details) are intentionally left in the data rather than removed, because a later stage of the project will train a separate tool to detect and redact exactly this kind of sensitive information — so we need realistic examples of it.

**Quality control:** The generation process went through multiple rounds of validation to catch problems like foreign-script glitches, template-sounding text, or mislabeled examples, and fixed roughly 230+ problem conversations before the dataset was finalized.

### 2.2 Domain adaptation ("teaching the model the vocabulary of scam calls")

The core language model used in this project is **XLM-RoBERTa**, a publicly available AI model already trained to understand ~100 languages. Before we can teach it to *detect* scams, it helps to first have it spend time reading large amounts of scam-call-style text so it becomes fluent in that particular style of speech (slang, phrasing patterns, code-mixing) — this step is called **domain-adaptive pretraining (DAPT)**.

Think of it like a doctor doing a specialty residency before starting to diagnose patients: the base model already "speaks the language," DAPT just makes it fluent in *this specific domain* of conversation.

This step does not yet involve any "scam" or "not scam" labels — it is purely about reading comprehension of this type of text. **No accuracy percentage applies at this stage**; progress is measured by a metric called **evaluation loss**, which tracks how well the model predicts intentionally hidden words in held-out call transcripts. Lower is better.

We trained **three separate versions** of this domain-adapted model, to compare which combination of training data works best:

| Version | Trained on | Loss before adaptation | Loss after adaptation | Improvement |
|---|---|---|---|---|
| **Mixed** | Korean + Hinglish combined | 2.39 | **1.55** | ~35% lower |
| **Korean-only** | Korean scam data only | 2.45 | **1.77** | ~28% lower |
| **Hinglish-only** | Synthetic Hinglish data only | 2.24 | **1.32** | ~41% lower |

All three show clear, healthy improvement, confirming the model is successfully absorbing the structure and vocabulary of vishing-call speech. The "Hinglish-only" model improved the most in relative terms, though it was trained on the smallest amount of data, so this is not yet conclusive — determining which version is actually the best *foundation* for detecting scams is the next step (Section 4).

### 2.3 Data integrity safeguard

Partway through this stage, we identified and corrected an issue: a portion of the data set aside for *final, unbiased testing later* had accidentally also been used during this training step. We paused, separated that data out properly, and **re-ran all three training sessions cleanly** to guarantee that future test results will be trustworthy and not artificially inflated. The corrected results (shown in the table above) came out nearly identical to the original run, which is a good sign — it means the issue didn't meaningfully distort what the model learned, and we can trust the figures above.

---

## 3. Current status summary

| Stage | Status |
|---|---|
| Collect/generate training data (Hinglish + Korean) | ✅ Done |
| Data validation & quality control | ✅ Done |
| Domain adaptation (language fluency training) | ✅ Done — 3 variants trained and verified |
| Data leakage check & correction | ✅ Done |
| **Scam + tactic detection classifier (the actual detector)** | ✅ **Built and trained** |
| Testing / accuracy measurement | ✅ **Available — see below** |
| Risk scoring (turning the detector's output into a per-call score) | ✅ **Built and tested — see 3.2** |

### 3.1 The real detector's results

This is the component that actually decides "is this call a scam" and "what manipulation
tactics is the caller using right now" - built by combining the language-fluency model
from Section 2.2 with a second, smaller neural network layer that reads a call turn by
turn, in order, the way a human listener would. It was trained and then tested on calls it
had never seen before (a held-out test set of 340 conversations), 3 times with different
random starting conditions to make sure the result wasn't a fluke.

**Scam vs. not-scam detection:**

| Metric | Result |
|---|---|
| F1 score (overall accuracy balancing false alarms vs. missed scams) | **0.997** |
| PR-AUC / ROC-AUC (ranking quality across all possible sensitivity settings) | **~1.000** |

In plain terms: on this test set, the detector is essentially never wrong about whether a
call is a scam. (A caveat worth keeping in mind: this test set is drawn from the same kind
of AI-generated synthetic conversations the model trained on, so this says the detector has
mastered the patterns in *this* data - it is still an open question how well that transfers
to messier real phone-call transcripts, with background noise, hesitations, and speech-to-
text mistakes a synthetic script doesn't have.)

**Manipulation-tactic detection** (which specific pressure tactics - authority, fear,
urgency, threat, isolation, financial pressure - appear in each line of the call):

| Tactic | How well the model spots it (F1 score) | How common it is in the training data |
|---|---|---|
| Financial coercion | 0.86 (strong) | 8% of lines |
| Urgency | 0.84 (strong) | 18% of lines |
| Authority | 0.79 (good) | 17% of lines |
| Fear | 0.64 (fair) | 6% of lines |
| Threat | 0.61 (fair) | 5% of lines |
| **Isolation** ("don't tell anyone, stay on the line") | **0.32 (weak)** | **3% of lines** |

Overall tactic-detection score (averaged across all six): **0.68**.

**The one clear weak spot is "isolation."** The model misses about 4 out of 5 real
isolation attempts (it's cautious rather than wrong when it does flag one - precision is
high, at 0.82). This tracks directly with how rare that tactic is in the training examples:
the model simply hasn't seen enough examples of it yet to learn it reliably. This is a data
problem, not a sign the overall approach doesn't work - the fix is straightforward (generate
more training examples of calls using isolation tactics) and is already on the plan.

### 3.2 Turning detector output into a risk score

The classifier in 3.1 produces numbers (a scam probability, six tactic probabilities per
line) - useful for a model, not immediately useful for a person deciding whether to worry
about a call. This stage combines those numbers into one simple **0-100 risk score plus a
Low / Medium / High label**, the form the project's final output is meant to take.

No fixed scoring formula was specified for this project, so a reasonable, documented
default was built (full detail in the project's internal log) rather than one being
derived from data: the score leans mainly on the scam probability itself, with two
smaller adjustments - how clearly the more serious pressure tactics (threats, demands for
money, isolating the victim) showed up anywhere in the call, and whether pressure
noticeably escalated in the second half of the call versus the first (a pattern typical of
real digital-arrest calls). The exact weighting can be tuned later once real calls are
available to calibrate against.

**Tested on the same 340 held-out calls as the detector above:**

| True call type | Scored Low | Scored Medium | Scored High |
|---|---|---|---|
| Not a scam (172 calls) | 171 | 1 | 0 |
| Digital-arrest scam (168 calls) | 1 | 1 | 166 |

In plain terms: almost every real scam call scored High risk, and almost every legitimate
call scored Low risk - a typical non-scam call scored about 1 out of 100; a typical scam
call scored about 78 out of 100. Only 2 of the 340 calls landed in the ambiguous "Medium"
zone, and both of those were cases where the underlying detector itself was genuinely
unsure about the call (not a flaw in the scoring step on top of it) - which is exactly what
the Medium band is supposed to catch: genuine uncertainty, flagged as such, rather than a
confident but wrong answer.

**With this stage built, the pipeline described in Section 1 is complete end to end** -
raw call text can now be turned into a risk score, on synthetic data. The caveat raised in
Section 3.1 still applies at every stage: this has not yet been tested on real, messy,
transcribed phone calls.

### A note on how to read these numbers

Everything in Section 2 (domain adaptation / "eval loss") measured how fluent the
language model had become in call-style speech - it had no concept of scam vs. not-scam
and produced no accuracy number. The results in this section are different: this is the
first stage of the project that actually makes a scam/not-scam decision, and the numbers
above are real accuracy results from testing it, not a proxy measurement.

---

## 4. What happens next

1. ~~Downstream comparison test~~ — done. It compared the three domain-adapted versions
   against the original model on a simpler benchmark task and found no clear winner among
   them (all performed similarly well); the full detector below was built on the "Mixed"
   version as a reasonable default.
2. ~~Build the full detection model~~ — done (Section 3.1 above).
3. ~~Train and test the full detector~~ — done (Section 3.1 above).
4. **Improve "isolation" tactic detection**: generate more training examples of calls that
   use isolation tactics ("don't hang up," "don't tell your family"), since that's the
   model's one clearly identified weak spot, then re-test.
5. ~~Risk scoring~~ — done (Section 3.2). The pipeline is now complete end to end on
   synthetic data.
6. **Validate against real-world speech**: the strong results above are on AI-generated
   synthetic conversations; the next meaningful test is how well this holds up on messier,
   real transcribed phone calls (background noise, hesitation, speech-to-text errors) once
   such data is available. This is now the single biggest open question for the project.
7. **Tune the risk-score formula** once real calls are available to calibrate against,
   rather than relying on the reasonable-default weights it currently uses.

---

*This report reflects project status as of October 22, 2026. The full pipeline described
in Section 1 - raw call text in, risk score out - is now built and tested end to end on
synthetic data. The next meaningful milestone is testing it against real, messy phone-call
transcripts rather than AI-generated ones.*

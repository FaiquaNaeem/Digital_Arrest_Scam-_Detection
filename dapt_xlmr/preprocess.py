"""
preprocess.py - load every configured source, normalise it into the common schema,
clean transcription artefacts, redact identifiers and remove duplicates.

Output (paths from config.paths):
  data/processed/documents.jsonl          one cleaned conversation per line
  reports/preprocessing_report.{json,md}  what was changed / removed and why
  reports/redaction_report.json           counts per identifier type (never the values)
  reports/removed_records.jsonl           every dropped record or document with its reason

Nothing is translated, and conversational content (threats, urgency, repetitions,
amounts of money) is never edited: the cleaner only touches encoding artefacts,
transcription markup, whitespace and the identifier patterns you enable.

Run:  python preprocess.py --config config.yaml
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import dapt_utils as du

LOGGER = logging.getLogger("dapt.preprocess")

# Zero-width characters that are pure encoding noise. ZWJ (U+200D) and ZWNJ (U+200C) are kept on
# purpose: they change Devanagari conjunct rendering and emoji sequences.
ZERO_WIDTH_RE = re.compile("[\u200b\u2060\ufeff\u00ad\u180e]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
NEWLINES_RE = re.compile("\r\n|\r|\u2028|\u2029|\x85")
HSPACE_RE = re.compile("[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")

_DIGIT_TABLE: dict[int, int] | None = None


def ascii_digit_table() -> dict[int, int]:
    """Map every non-ASCII Unicode decimal digit (Devanagari, full-width, ...) to ASCII."""
    global _DIGIT_TABLE
    if _DIGIT_TABLE is None:
        table = {}
        for cp in range(sys.maxunicode + 1):
            ch = chr(cp)
            if ch.isdecimal() and not (0x30 <= cp <= 0x39):
                try:
                    table[cp] = 0x30 + unicodedata.decimal(ch)
                except ValueError:
                    continue
        _DIGIT_TABLE = table
    return _DIGIT_TABLE


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #
def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Order matters: earlier detectors win, so specific formats run before generic digit runs.
# Lookarounds use explicit [0-9] instead of \b because Hangul letters are "word" characters
# ("010-1234-5678로" has no \b after the 8).
BUILTIN_DETECTORS: list[tuple[str, str, str]] = [
    ("email", "EMAIL", r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])"),
    ("url", "URL", r"(?:https?://|www\.)[^\s<>\"]+"),
    ("upi_id", "UPI_ID", r"(?<![\w.@-])[A-Za-z0-9._-]{2,256}@[A-Za-z]{2,64}(?![\w.@-])"),
    ("kr_rrn", "KR_RRN", r"(?<![0-9])[0-9]{6}\s?-\s?[1-8][0-9]{6}(?![0-9])"),
    ("card", "CARD_NUMBER", r"(?<![0-9])(?:[0-9][ -]?){12,18}[0-9](?![0-9])"),
    ("phone_kr", "PHONE",
     r"(?<![0-9])(?:(?:\+|00)82[ -]?(?:0)?1[016789]|01[016789])[ -]?[0-9]{3,4}[ -]?[0-9]{4}(?![0-9])"
     r"|(?<![0-9])0(?:2|[3-6][1-5])[ -]?[0-9]{3,4}[ -]?[0-9]{4}(?![0-9])"),
    ("phone_in", "PHONE", r"(?<![0-9])(?:(?:\+|00)?91[ -]?|0)?[6-9][0-9]{4}[ -]?[0-9]{5}(?![0-9])"),
    # Aadhaar after phones (so "+91 98765 43210" stays a phone) and never inside a longer digit group.
    ("aadhaar", "AADHAAR_NUMBER",
     r"(?<![0-9])(?<![0-9][ -])[2-9][0-9]{3}[ -]?[0-9]{4}[ -]?[0-9]{4}(?![0-9])(?![ -][0-9])"),
    ("pan", "PAN_NUMBER", r"(?<![A-Za-z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Za-z0-9])"),
    ("ifsc", "IFSC", r"(?<![A-Za-z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Za-z0-9])"),
    ("bank_account", "ACCOUNT_NUMBER", r"(?<![0-9])[0-9]{9,18}(?![0-9])"),
]


class Redactor:
    """Regex-based identifier redaction. This is masking of obvious patterns, not anonymisation:
    names, addresses and spelled-out numbers are not detected."""

    def __init__(self, rcfg: dict):
        self.enabled = bool(rcfg.get("enabled", True))
        fmt = rcfg.get("placeholder_format", "[{label}]")
        enabled_types = rcfg.get("types") or {}
        self.detectors: list[tuple[str, str, re.Pattern]] = []
        if not self.enabled:
            return
        for key, label, pattern in BUILTIN_DETECTORS:
            if enabled_types.get(key, False):
                self.detectors.append((key, fmt.format(label=label), re.compile(pattern)))
        for i, custom in enumerate(rcfg.get("custom_patterns") or []):
            name = custom.get("name") or f"custom_{i}"
            flags = re.IGNORECASE if "i" in str(custom.get("flags", "")) else 0
            label = custom.get("label", name.upper())
            self.detectors.append((name, fmt.format(label=label), re.compile(custom["pattern"], flags)))

    def __call__(self, text: str) -> tuple[str, Counter]:
        counts: Counter = Counter()
        if not self.enabled:
            return text, counts
        for key, placeholder, regex in self.detectors:
            def _sub(m: re.Match, key=key, placeholder=placeholder) -> str:
                if key == "card" and not _luhn_ok(m.group(0)):
                    return m.group(0)
                counts[key] += 1
                return placeholder
            text = regex.sub(_sub, text)
        return text, counts


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
class TextCleaner:
    """Deterministic text normalisation shared by DAPT preprocessing and downstream data."""

    def __init__(self, pcfg: dict):
        self.unicode_form = pcfg.get("unicode_form", "NFC")
        if self.unicode_form not in ("NFC", "NFKC", "none"):
            raise ValueError("preprocessing.unicode_form must be NFC, NFKC or none")
        self.ascii_digits = bool(pcfg.get("normalize_digits_to_ascii", True))
        self.remove_zero_width = bool(pcfg.get("remove_zero_width", True))
        self.strip_control = bool(pcfg.get("strip_control_chars", True))
        self.max_newlines = int(pcfg.get("max_consecutive_newlines", 2))
        self.artifacts = []
        for i, art in enumerate(pcfg.get("artifact_patterns") or []):
            flags = re.MULTILINE | (re.IGNORECASE if "i" in str(art.get("flags", "")) else 0)
            self.artifacts.append((art.get("name") or f"artifact_{i}", re.compile(art["pattern"], flags),
                                   art.get("replacement", "")))
        spk = pcfg.get("speaker_labels") or {}
        self.speaker_mode = spk.get("mode", "neutralize")
        if self.speaker_mode not in ("keep", "neutralize", "remove"):
            raise ValueError("preprocessing.speaker_labels.mode must be keep|neutralize|remove")
        self.speaker_in_free_text = bool(spk.get("apply_to_free_text", False))
        self.free_text_speaker_re = re.compile(
            spk.get("free_text_pattern", r"^(\s*)([^\n:：]{1,24}?)\s*[:：][ \t]*"), re.MULTILINE)
        self.redactor = Redactor(pcfg.get("redaction") or {})

    # -- speaker handling ------------------------------------------------------
    def format_turns(self, turns: list[tuple[str | None, str]]) -> tuple[str, int]:
        """Join structured turns. 'neutralize' maps each distinct speaker to S1, S2, ... in order of
        appearance, so role names such as "scammer"/"victim" never leak a label into the text."""
        mapping: dict[str, str] = {}
        lines = []
        for speaker, text in turns:
            text = text.strip()
            if speaker is None or self.speaker_mode == "remove":
                lines.append(text)
            elif self.speaker_mode == "keep":
                lines.append(f"{speaker}: {text}")
            else:
                key = speaker.strip().casefold()
                if key not in mapping:
                    mapping[key] = f"S{len(mapping) + 1}"
                lines.append(f"{mapping[key]}: {text}")
        return "\n".join(lines), len(mapping)

    def _free_text_speakers(self, text: str) -> tuple[str, int]:
        mapping: dict[str, str] = {}

        def _sub(m: re.Match) -> str:
            if self.speaker_mode == "remove":
                return m.group(1)
            key = m.group(2).strip().casefold()
            if key not in mapping:
                mapping[key] = f"S{len(mapping) + 1}"
            return f"{m.group(1)}{mapping[key]}: "

        if self.speaker_mode == "keep":
            return text, 0
        return self.free_text_speaker_re.sub(_sub, text), len(mapping)

    # -- main entry ------------------------------------------------------------
    def basic_normalize(self, text: str) -> str:
        """Unicode + whitespace normalisation only (used for inference inputs)."""
        if self.unicode_form != "none":
            text = unicodedata.normalize(self.unicode_form, text)
        text = NEWLINES_RE.sub("\n", text)
        return self._whitespace(text)

    def _whitespace(self, text: str) -> str:
        text = HSPACE_RE.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        if self.max_newlines > 0:
            text = re.sub(r"\n{%d,}" % (self.max_newlines + 1), "\n" * self.max_newlines, text)
        return text.strip()

    def clean(self, text: str) -> tuple[str, Counter]:
        stats: Counter = Counter()
        if self.unicode_form != "none":
            normalised = unicodedata.normalize(self.unicode_form, text)
            if normalised != text:
                stats["unicode_normalised_docs"] += 1
            text = normalised
        if self.ascii_digits:
            converted = text.translate(ascii_digit_table())
            if converted != text:
                stats["non_ascii_digits_converted_docs"] += 1
            text = converted
        text = NEWLINES_RE.sub("\n", text)
        if self.remove_zero_width:
            text, n = ZERO_WIDTH_RE.subn("", text)
            stats["zero_width_chars_removed"] += n
        if self.strip_control:
            text, n = CONTROL_RE.subn("", text)
            stats["control_chars_removed"] += n
        for name, regex, replacement in self.artifacts:
            text, n = regex.subn(replacement, text)
            stats[f"artifact:{name}"] += n
        if self.speaker_in_free_text:
            text, n = self._free_text_speakers(text)
            stats["free_text_speakers_mapped"] += n
        text = self._whitespace(text)
        text, red = self.redactor(text)
        for key, n in red.items():
            stats[f"redacted:{key}"] += n
        text = self._whitespace(text)
        return text, stats


# --------------------------------------------------------------------------- #
# Near-duplicate detection (MinHash + LSH, numpy only)
# --------------------------------------------------------------------------- #
def _lsh_shape(num_perm: int, threshold: float) -> tuple[int, int]:
    best = None
    for rows in range(1, num_perm + 1):
        if num_perm % rows:
            continue
        bands = num_perm // rows
        approx = (1.0 / bands) ** (1.0 / rows)
        score = abs(approx - threshold)
        if best is None or score < best[0]:
            best = (score, bands, rows)
    return best[1], best[2]


def find_near_duplicates(docs: list[dict], threshold: float = 0.9, num_perm: int = 128,
                         shingle_size: int = 5, seed: int = 0) -> dict[int, tuple[int, float]]:
    """Return {doc_position: (kept_doc_position, estimated_jaccard)} for near-duplicates.

    Character shingles work for Hangul and romanised Hinglish without a word tokenizer.
    The earliest document of each cluster (in source order) is kept.
    """
    import hashlib

    import numpy as np

    prime = np.uint64(4294967311)  # smallest prime > 2**32
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**32 - 1, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, 2**32 - 1, size=num_perm, dtype=np.uint64)
    bands, rows = _lsh_shape(num_perm, threshold)
    buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(bands)]
    signatures: dict[int, np.ndarray] = {}
    duplicates: dict[int, tuple[int, float]] = {}

    for pos, doc in enumerate(docs):
        norm = " ".join(doc["text"].casefold().split())
        if len(norm) <= shingle_size:
            shingles = {norm}
        else:
            shingles = {norm[i:i + shingle_size] for i in range(len(norm) - shingle_size + 1)}
        hashes = np.fromiter(
            (int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little") for s in shingles),
            dtype=np.uint64, count=len(shingles))
        # (a*x) < 2**64 because a, x < 2**32, so uint64 arithmetic is exact before the modulo.
        sig = ((((a[:, None] * hashes[None, :]) % prime) + b[:, None]) % prime).min(axis=1)
        candidates = set()
        keys = []
        for band in range(bands):
            key = sig[band * rows:(band + 1) * rows].tobytes()
            keys.append(key)
            candidates.update(buckets[band].get(key, ()))
        best = None
        for other in candidates:
            est = float((signatures[other] == sig).mean())
            if est >= threshold and (best is None or est > best[1]):
                best = (other, est)
        if best is not None:
            duplicates[pos] = best
            continue
        signatures[pos] = sig
        for band, key in enumerate(keys):
            buckets[band][key].append(pos)
    return duplicates


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_preprocessing(cfg: dict, sources: list[dict] | None = None, keep_labels: bool = False,
                      write_outputs: bool = True, report_name: str = "preprocessing",
                      max_records_per_source: int | None = None) -> tuple[list[dict], dict]:
    """Load + clean + deduplicate. Returns (documents, report).

    keep_labels=False (DAPT) drops every configured label field; their presence is only counted.
    keep_labels=True is used by the downstream (supervised) data path.
    """
    pcfg = cfg.get("preprocessing") or {}
    sources = sources if sources is not None else cfg["sources"]
    label_fields = cfg.get("label_fields") or []
    cleaner = TextCleaner(pcfg)
    reports_dir = du.resolve_path(cfg, du.cfg_get(cfg, "paths.reports_dir", "reports"))
    removed: list[dict] = []
    file_errors: list[dict] = []
    per_source: dict[str, Counter] = {}
    labels_seen: dict[str, Counter] = defaultdict(Counter)
    notes_total: Counter = Counter()
    clean_stats: dict[str, Counter] = defaultdict(Counter)
    redaction_docs: dict[str, Counter] = defaultdict(Counter)
    docs: list[dict] = []
    missing_required = []

    for source in sources:
        name = source["name"]
        stats: Counter = Counter()
        per_source[name] = stats
        files, missing = du.discover_source_files(cfg, source)
        stats["files"] = len(files)
        if missing:
            msg = f"source {name!r}: not found -> {missing}"
            if source.get("required", True):
                missing_required.append(msg)
            else:
                LOGGER.warning("%s (source marked required: false, skipping)", msg)
        units = []
        for path in files:
            for item in du.iter_file_records(cfg, source, path):
                if "error" in item:
                    if item["index"] == -1:
                        file_errors.append({"source": name, "file": item["file"], "error": item["error"]})
                        stats["file_errors"] += 1
                    else:
                        stats["record_errors"] += 1
                    removed.append({"source": name, "category": source["category"], "ref": f"{item['file']}#{item['index']}",
                                    "reason": item["error"].split(":")[0]})
                    continue
                stats["records_read"] += 1
                if max_records_per_source and stats["records_read"] > max_records_per_source:
                    stats["records_skipped_by_limit"] += 1
                    continue
                unit, reason = du.record_to_unit(item, source, label_fields)
                if unit is None:
                    stats[f"removed:{reason}"] += 1
                    removed.append({"source": name, "category": source["category"],
                                    "ref": f"{item['file']}#{item['index']}", "reason": reason})
                    continue
                for field in unit["labels"]:
                    labels_seen[source["category"]][field] += 1
                units.append(unit)
        assembled, notes = du.assemble_documents(units, assemble=bool(source.get("assemble_by_conversation", True)))
        notes_total.update({f"{name}:{k}": v for k, v in notes.items()})
        stats["documents_assembled"] = len(assembled)

        for doc in assembled:
            raw_text, n_speakers = cleaner.format_turns(doc.pop("turns"))
            text, cstats = cleaner.clean(raw_text)
            cat = doc["category"]
            clean_stats[cat].update(cstats)
            for key in cstats:
                if key.startswith("redacted:"):
                    redaction_docs[cat][key.split(":", 1)[1]] += 1
            min_chars = int(pcfg.get("min_chars", 1))
            if len(text) < min_chars:
                reason = "empty_after_cleaning" if not text else f"shorter_than_min_chars_{min_chars}"
                stats[f"removed:{reason}"] += 1
                removed.append({"source": name, "category": cat, "ref": doc["doc_id"], "reason": reason})
                continue
            doc["text"] = text
            doc["text_sha1"] = du.sha1_text(text)
            doc["n_chars"] = len(text)
            doc["n_speakers_mapped"] = n_speakers
            if not keep_labels:
                doc["label_fields_excluded"] = sorted(doc.pop("labels").keys())
                doc.pop("label_conflicts", None)
            docs.append(doc)
            stats["documents_kept_after_cleaning"] += 1

    if missing_required:
        raise FileNotFoundError(
            "Required data sources are missing (fix the paths in config.yaml or set `required: false`):\n  "
            + "\n  ".join(missing_required))
    if file_errors and pcfg.get("fail_on_file_errors", True):
        lines = "\n  ".join(f"{e['file']}: {e['error']}" for e in file_errors)
        raise RuntimeError(f"{len(file_errors)} file(s) could not be read:\n  {lines}\n"
                           "Fix them, or set preprocessing.fail_on_file_errors: false to skip (they will be "
                           "listed in the report).")

    # Exact duplicates (global, first occurrence wins).
    dedup_cfg = pcfg.get("dedup") or {}
    exact_removed: Counter = Counter()
    cross_category: Counter = Counter()
    if dedup_cfg.get("exact", True):
        first_seen: dict[str, dict] = {}
        kept = []
        for doc in docs:
            h = doc["text_sha1"]
            if h in first_seen:
                orig = first_seen[h]
                exact_removed[doc["category"]] += 1
                if orig["category"] != doc["category"]:
                    cross_category[f"{orig['category']}<->{doc['category']}"] += 1
                removed.append({"source": doc["source_name"], "category": doc["category"], "ref": doc["doc_id"],
                                "reason": "exact_duplicate", "duplicate_of": orig["doc_id"]})
                continue
            first_seen[h] = doc
            kept.append(doc)
        docs = kept

    near_cfg = dedup_cfg.get("near_duplicate") or {}
    near_removed: Counter = Counter()
    if near_cfg.get("enabled", False):
        LOGGER.info("Running near-duplicate detection on %d documents ...", len(docs))
        dups = find_near_duplicates(docs, float(near_cfg.get("threshold", 0.9)), int(near_cfg.get("num_perm", 128)),
                                    int(near_cfg.get("shingle_size", 5)), int(du.cfg_get(cfg, "project.seed", 42)))
        kept = []
        for pos, doc in enumerate(docs):
            if pos in dups:
                other, est = dups[pos]
                near_removed[doc["category"]] += 1
                removed.append({"source": doc["source_name"], "category": doc["category"], "ref": doc["doc_id"],
                                "reason": "near_duplicate", "duplicate_of": docs[other]["doc_id"],
                                "estimated_jaccard": round(est, 3)})
                continue
            kept.append(doc)
        docs = kept

    # ---------------- report ----------------
    by_cat: dict[str, dict] = {}
    for doc in docs:
        c = by_cat.setdefault(doc["category"], {"documents": 0, "chars": 0, "languages": Counter(),
                                                 "source_types": Counter(), "records": 0})
        c["documents"] += 1
        c["chars"] += doc["n_chars"]
        c["records"] += doc["n_records"]
        c["languages"][doc["language"]] += 1
        c["source_types"][doc["source_type"]] += 1
    removed_by_reason: dict[str, Counter] = defaultdict(Counter)
    for r in removed:
        removed_by_reason[r["category"]][r["reason"]] += 1

    report = {
        "created": du.now_iso(),
        "documents_kept": len(docs),
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "per_category": {k: {**v, "languages": dict(v["languages"]), "source_types": dict(v["source_types"])}
                         for k, v in by_cat.items()},
        "removed_by_category_and_reason": {k: dict(v) for k, v in removed_by_reason.items()},
        "exact_duplicates_removed": dict(exact_removed),
        "cross_category_exact_duplicates": dict(cross_category),
        "near_duplicates_removed": dict(near_removed) if near_cfg.get("enabled", False) else "disabled",
        "cleaning_counts": {k: dict(v) for k, v in clean_stats.items()},
        "label_fields_found": {k: dict(v) for k, v in labels_seen.items()},
        "label_fields_policy": "kept for downstream use" if keep_labels else
                               "excluded from DAPT documents (never part of text or metadata)",
        "assembly_notes": dict(notes_total),
        "file_errors": file_errors,
        "speaker_label_mode": cleaner.speaker_mode,
    }
    redaction_report = {
        "enabled": cleaner.redactor.enabled,
        "detectors": [d[0] for d in cleaner.redactor.detectors],
        "replacements_by_category": {c: {k.split(":", 1)[1]: v for k, v in s.items() if k.startswith("redacted:")}
                                     for c, s in clean_stats.items()},
        "documents_affected_by_category": {k: dict(v) for k, v in redaction_docs.items()},
        "note": "Counts only; redacted values are never written anywhere. Regex redaction does not detect "
                "names, addresses or spoken-out numbers.",
    }
    report["redaction"] = redaction_report

    for w in _report_warnings(report):
        LOGGER.warning(w)
    report["warnings"] = _report_warnings(report)

    if write_outputs:
        processed_dir = du.resolve_path(cfg, du.cfg_get(cfg, "paths.processed_dir", "data/processed"))
        du.write_jsonl(processed_dir / "documents.jsonl", docs)
        du.write_json(reports_dir / f"{report_name}_report.json", report)
        du.write_json(reports_dir / "redaction_report.json", redaction_report)
        du.write_jsonl(reports_dir / "removed_records.jsonl", removed)
        du.write_text(reports_dir / f"{report_name}_report.md", render_markdown(report))
        LOGGER.info("Wrote %d documents to %s", len(docs), processed_dir / "documents.jsonl")
        LOGGER.info("Reports in %s", reports_dir)
    return docs, report


def _report_warnings(report: dict) -> list[str]:
    out = []
    if report["cross_category_exact_duplicates"]:
        out.append(f"Identical conversations found in different categories: "
                   f"{report['cross_category_exact_duplicates']} - check your source folders.")
    for cat, fields in report["label_fields_found"].items():
        if fields and report["label_fields_policy"].startswith("excluded"):
            out.append(f"{cat}: label fields {sorted(fields)} were present and excluded from DAPT data.")
    for key, n in report["assembly_notes"].items():
        if key.endswith("label_conflict_within_conversation") and n:
            out.append(f"{key}: {n} conversation(s) have conflicting labels across their records.")
    return out


def render_markdown(report: dict) -> str:
    lines = [f"# Preprocessing report\n\nCreated {report['created']}. Documents kept: **{report['documents_kept']:,}**\n"]
    lines.append("## Per source\n")
    keys = sorted({k for v in report["per_source"].values() for k in v})
    lines.append(du.md_table(["source"] + keys, [[s] + [v.get(k, 0) for k in keys]
                                                 for s, v in report["per_source"].items()]))
    lines.append("\n## Per category (after cleaning and deduplication)\n")
    lines.append(du.md_table(["category", "documents", "records", "chars", "languages", "source types"],
                             [[c, v["documents"], v["records"], v["chars"], v["languages"], v["source_types"]]
                              for c, v in report["per_category"].items()]))
    lines.append("\n## Removed (reason counts)\n")
    rows = [[c, r, n] for c, reasons in report["removed_by_category_and_reason"].items() for r, n in reasons.items()]
    lines.append(du.md_table(["category", "reason", "count"], rows) if rows else "Nothing removed.")
    lines.append("\nEvery removed item is listed in `removed_records.jsonl` (ids and reasons, no text).\n")
    lines.append("## Redactions (counts only)\n")
    red = report["redaction"]
    rows = [[c, k, n] for c, d in red["replacements_by_category"].items() for k, n in d.items()]
    lines.append(f"Enabled: {red['enabled']}; detectors: {', '.join(red['detectors']) or 'none'}\n")
    lines.append(du.md_table(["category", "type", "replacements"], rows) if rows else "No identifiers matched.")
    lines.append("\n## Cleaning counts\n")
    rows = [[c, k, n] for c, d in report["cleaning_counts"].items() for k, n in sorted(d.items())
            if not k.startswith("redacted:") and n]
    lines.append(du.md_table(["category", "operation", "count"], rows) if rows else "No changes.")
    lines.append(f"\n## Labels\n\nPolicy: {report['label_fields_policy']}. Found: {report['label_fields_found']}\n")
    if report["warnings"]:
        lines.append("## Warnings\n")
        lines += [f"- {w}" for w in report["warnings"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Override any config value, e.g. --set preprocessing.redaction.enabled=false")
    parser.add_argument("--max_records_per_source", type=int, default=None,
                        help="Only read the first N records of each source (quick preview).")
    args = parser.parse_args()
    cfg = du.load_config(args.config, args.overrides)
    du.setup_logging(cfg, "preprocess")
    docs, report = run_preprocessing(cfg, max_records_per_source=args.max_records_per_source)
    for cat, v in report["per_category"].items():
        LOGGER.info("%-16s documents=%-7d chars=%d", cat, v["documents"], v["chars"])


if __name__ == "__main__":
    main()

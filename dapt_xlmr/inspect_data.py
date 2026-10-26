"""
inspect_data.py - look at the raw sources before anything is changed.

For every configured source it reports: files found / missing, format, records, parse errors,
fields present, missing or empty text, conversation_id coverage, language and source_type
values, label fields present, raw length statistics, letter-script mix (to catch files filed
under the wrong language), candidate speaker labels at line starts (to spot role names such as
"scammer:" that would leak labels), and exact duplicate texts.

With --tokenizer it also measures XLM-R token counts and the <unk> rate on a sample.

Nothing is modified. Output: reports/inspection_report.{json,md}

Run:  python inspect_data.py --config config.yaml [--tokenizer FacebookAI/xlm-roberta-base]
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter, defaultdict

import dapt_utils as du

LOGGER = logging.getLogger("dapt.inspect")


def inspect_source(cfg: dict, source: dict, label_fields: list[str], sample_size: int) -> dict:
    files, missing = du.discover_source_files(cfg, source)
    text_field = source.get("text_field", "text")
    conv_field = source.get("conversation_id_field", "conversation_id")
    info: dict = {
        "name": source["name"], "category": source["category"], "required": source.get("required", True),
        "files_found": len(files), "missing_paths": missing, "formats": Counter(),
        "records": 0, "record_errors": Counter(), "file_errors": [], "fields": Counter(),
        "text_issues": Counter(), "with_conversation_id": 0, "languages": Counter(),
        "source_types": Counter(), "label_fields": Counter(), "label_values": defaultdict(Counter),
        "chars": [], "scripts": Counter(), "speaker_candidates": Counter(), "duplicate_texts": 0,
        "conversation_ids": set(),
    }
    seen_hashes = set()
    samples: list[str] = []
    rng = random.Random(0)
    for path in files:
        info["formats"][du.file_format(source, path)] += 1
        for item in du.iter_file_records(cfg, source, path):
            if "error" in item:
                if item["index"] == -1:
                    info["file_errors"].append({"file": item["file"], "error": item["error"]})
                else:
                    info["record_errors"][item["error"].split(":")[0]] += 1
                continue
            rec = item["record"]
            info["records"] += 1
            info["fields"].update(rec.keys())
            for lf in label_fields:
                if lf in rec and rec[lf] not in (None, ""):
                    info["label_fields"][lf] += 1
                    info["label_values"][lf][str(rec[lf])[:40]] += 1
            unit, reason = du.record_to_unit(item, source, label_fields)
            if unit is None:
                info["text_issues"][reason] += 1
                continue
            if unit["conversation_id"] is not None:
                info["with_conversation_id"] += 1
                info["conversation_ids"].add(unit["conversation_id"])
            info["languages"][unit["language"]] += 1
            info["source_types"][unit["source_type"]] += 1
            text = "\n".join(t for _, t in unit["turns"])
            info["chars"].append(len(text))
            info["scripts"].update(du.script_profile(text))
            info["speaker_candidates"].update(du.speaker_label_candidates(text))
            for s, _ in unit["turns"]:
                if s:
                    info["speaker_candidates"][f"[speaker field] {s}"] += 1
            h = du.sha1_text(text.strip())
            if h in seen_hashes:
                info["duplicate_texts"] += 1
            seen_hashes.add(h)
            if len(samples) < sample_size:
                samples.append(text)
            else:
                j = rng.randrange(info["records"])
                if j < sample_size:
                    samples[j] = text
    info["n_conversation_ids"] = len(info.pop("conversation_ids"))
    info["_samples"] = samples
    if text_field not in info["fields"] and not source.get("turns_field") and info["records"]:
        info["text_issues"][f"text_field_{text_field!r}_never_present"] += 1
    info["conversation_id_field"] = conv_field
    return info


def script_shares(scripts: Counter) -> dict:
    letters = sum(v for k, v in scripts.items() if k != "digit") or 1
    return {k: round(v / letters, 3) for k, v in scripts.items() if k != "digit"}


def tokenizer_stats(tokenizer, texts: list[str]) -> dict:
    if not texts:
        return {}
    counts, unk, total = [], 0, 0
    unk_id = tokenizer.unk_token_id
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        counts.append(len(ids))
        total += len(ids)
        unk += sum(1 for i in ids if i == unk_id)
    return {"sampled_texts": len(texts), "tokens_per_text": du.percentiles(counts),
            "mean_tokens_per_text": sum(counts) / len(counts), "unk_rate": unk / max(1, total),
            "chars_per_token": sum(len(t) for t in texts) / max(1, total)}


def render_markdown(report: dict) -> str:
    lines = [f"# Data inspection report\n\nCreated {report['created']}. Nothing was modified.\n"]
    rows = []
    for s in report["sources"]:
        rows.append([s["name"], s["category"], s["files_found"], s["records"], sum(s["record_errors"].values()),
                     len(s["file_errors"]), sum(s["text_issues"].values()), s["with_conversation_id"],
                     s["n_conversation_ids"], s["duplicate_texts"]])
    lines.append(du.md_table(["source", "category", "files", "records", "bad records", "bad files",
                              "missing/empty text", "with conv id", "distinct conv ids", "exact dup texts"], rows))
    for s in report["sources"]:
        lines.append(f"\n## {s['name']} ({s['category']})\n")
        if s["missing_paths"]:
            lines.append(f"- **Missing paths:** {s['missing_paths']}")
        lines.append(f"- Formats: {dict(s['formats'])}")
        lines.append(f"- Fields seen (top 20): {dict(Counter(s['fields']).most_common(20))}")
        lines.append(f"- Text issues: {dict(s['text_issues']) or 'none'}")
        lines.append(f"- Record errors: {dict(s['record_errors']) or 'none'}; file errors: {s['file_errors'] or 'none'}")
        lines.append(f"- Languages: {dict(s['languages'])}; source types: {dict(s['source_types'])}")
        lines.append(f"- Label fields present (excluded from DAPT): {dict(s['label_fields']) or 'none'}")
        lines.append(f"- Raw chars per record: {s['chars_percentiles']}")
        lines.append(f"- Letter script shares: {s['script_shares']}")
        lines.append(f"- Candidate speaker labels (top 15): {s['speaker_candidates_top']}")
        if s.get("tokenizer"):
            lines.append(f"- Tokenizer sample: {s['tokenizer']}")
    if report["warnings"]:
        lines.append("\n## Warnings\n")
        lines += [f"- {w}" for w in report["warnings"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--tokenizer", default=None, help="e.g. FacebookAI/xlm-roberta-base (needs transformers)")
    parser.add_argument("--sample_size", type=int, default=200, help="texts per source for tokenizer stats")
    parser.add_argument("--strict", action="store_true", help="exit with code 1 if a required source is missing")
    args = parser.parse_args()

    cfg = du.load_config(args.config, args.overrides)
    du.setup_logging(cfg, "inspect_data")
    label_fields = cfg.get("label_fields") or []
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer  # lazy: only needed for this option
        tokenizer = AutoTokenizer.from_pretrained(du.resolve_model_ref(cfg, args.tokenizer))

    report = {"created": du.now_iso(), "sources": [], "warnings": []}
    expected_scripts = {"korean": "hangul", "ko": "hangul"}
    for source in cfg["sources"]:
        LOGGER.info("Inspecting source %s ...", source["name"])
        info = inspect_source(cfg, source, label_fields, args.sample_size)
        info["chars_percentiles"] = du.percentiles(info.pop("chars"))
        info["script_shares"] = script_shares(info["scripts"])
        info["speaker_candidates_top"] = dict(info["speaker_candidates"].most_common(15))
        samples = info.pop("_samples")
        if tokenizer is not None:
            info["tokenizer"] = tokenizer_stats(tokenizer, samples)
            if info["records"] and info["tokenizer"]:
                info["tokenizer"]["estimated_total_tokens"] = int(info["tokenizer"]["mean_tokens_per_text"] * info["records"])
        name = source["name"]
        if info["missing_paths"]:
            report["warnings"].append(f"{name}: missing paths {info['missing_paths']}")
        if info["file_errors"]:
            report["warnings"].append(f"{name}: {len(info['file_errors'])} unreadable file(s)")
        if info["records"] and info["with_conversation_id"] == 0 and source.get("txt_mode", "file") != "file":
            report["warnings"].append(f"{name}: no conversation ids; every record is treated as its own conversation. "
                                      "If records are turns of one call, add a conversation_id field.")
        if info["label_fields"]:
            report["warnings"].append(f"{name}: label fields {dict(info['label_fields'])} present - they will be "
                                      "excluded from DAPT text and metadata.")
        lang = str(source.get("language", "")).lower()
        shares = info["script_shares"]
        if expected_scripts.get(lang) == "hangul" and shares and shares.get("hangul", 0) < 0.5:
            report["warnings"].append(f"{name}: declared Korean but only {shares.get('hangul', 0):.0%} of letters are Hangul.")
        if lang in ("hi-en", "hinglish", "hi_en") and shares.get("hangul", 0) > 0.05:
            report["warnings"].append(f"{name}: declared Hinglish but {shares['hangul']:.0%} of letters are Hangul.")
        suspicious = [k for k in info["speaker_candidates"] if any(
            w in k.casefold() for w in ("scam", "fraud", "victim", "사기", "피해", "범인", "thug", "fake"))]
        if suspicious:
            report["warnings"].append(f"{name}: speaker labels that reveal the class: {suspicious[:5]}. Use "
                                      "preprocessing.speaker_labels.mode=neutralize (and apply_to_free_text for "
                                      "plain-text transcripts).")
        info["record_errors"] = dict(info["record_errors"])
        info["label_values"] = {k: dict(v.most_common(10)) for k, v in info["label_values"].items()}
        info.pop("speaker_candidates")
        report["sources"].append(info)

    reports_dir = du.resolve_path(cfg, du.cfg_get(cfg, "paths.reports_dir", "reports"))
    du.write_json(reports_dir / "inspection_report.json", report)
    du.write_text(reports_dir / "inspection_report.md", render_markdown(report))
    for w in report["warnings"]:
        LOGGER.warning(w)
    for s in report["sources"]:
        LOGGER.info("%-22s %-14s files=%-4d records=%-7d text-issues=%s", s["name"], s["category"],
                    s["files_found"], s["records"], dict(s["text_issues"]) or 0)
    LOGGER.info("Report: %s", reports_dir / "inspection_report.md")
    missing_required = [s for s in report["sources"] if s["missing_paths"] and s["required"]]
    if args.strict and missing_required:
        sys.exit(1)


if __name__ == "__main__":
    main()

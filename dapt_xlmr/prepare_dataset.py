"""
prepare_dataset.py - build leakage-free train / validation / test splits for DAPT.

Steps
  1. run preprocess.py's pipeline (load all formats -> common schema -> clean -> dedup)
     or reuse data/processed/documents.jsonl with --use_existing_processed
  2. drop conversations listed in `exclusions` (e.g. your downstream test set)
  3. give every document a leakage group (conversation_id, or e.g. recording_id)
  4. split *groups* (never chunks) per category with a fixed seed;
     source types listed in split.train_only_source_types (e.g. synthetic) stay in train
  5. verify that no group and no identical text appears in two splits
  6. write data/prepared/{train,validation,test}.jsonl, manifest.json and dataset statistics

Tokenisation and chunking happen later (train_dapt.py), i.e. strictly after splitting,
so chunks of one call can never end up on both sides of the split.

Run:  python prepare_dataset.py --config config.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from collections import Counter, defaultdict

import dapt_utils as du
import preprocess

LOGGER = logging.getLogger("dapt.prepare")
SPLITS = ("train", "validation", "test")
DOC_FIELDS = ("doc_id", "group_id", "conversation_id", "conversation_id_generated", "category", "source_name",
              "language", "source_type", "source_files", "record_refs", "n_records", "text", "text_sha1",
              "n_chars", "metadata")


def load_exclusions(cfg: dict) -> tuple[set[str], set[str], dict]:
    """Conversation ids and cleaned-text hashes that must never enter the DAPT corpus."""
    ecfg = cfg.get("exclusions") or {}
    ids: set[str] = set()
    hashes: set[str] = set()
    info = {"conversation_id_files": [], "text_files": []}
    for f in ecfg.get("conversation_id_files") or []:
        path = du.resolve_path(cfg, f)
        if not path.is_file():
            raise FileNotFoundError(f"exclusions.conversation_id_files entry not found: {path}")
        new = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        ids |= new
        info["conversation_id_files"].append({"file": str(f), "ids": len(new)})
    text_files = ecfg.get("text_files") or []
    if text_files:
        source = {
            "name": "_exclusions", "category": "_exclusions", "paths": list(text_files),
            "text_field": ecfg.get("text_field", "text"), "turns_field": ecfg.get("turns_field"),
            "turn_text_field": ecfg.get("turn_text_field", "text"),
            "turn_speaker_field": ecfg.get("turn_speaker_field", "speaker"),
            "conversation_id_field": ecfg.get("conversation_id_field", "conversation_id"),
        }
        docs, _ = preprocess.run_preprocessing(cfg, sources=[source], write_outputs=False)
        for d in docs:
            hashes.add(d["text_sha1"])
            if not d["conversation_id_generated"]:
                ids.add(d["conversation_id"])
        info["text_files"].append({"files": list(text_files), "texts": len(docs)})
    return ids, hashes, info


def split_documents(cfg: dict, docs: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    scfg = cfg.get("split") or {}
    seed = int(du.cfg_get(cfg, "project.seed", 42))
    group_notes = du.assign_group_ids(docs, scfg.get("group_by", "conversation_id"),
                                      scfg.get("group_namespace", "source"))
    train_only = {str(t) for t in scfg.get("train_only_source_types") or []}
    groups_by_cat: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for d in docs:
        groups_by_cat[d["category"]][d["group_id"]].append(d)

    assignment: dict[str, str] = {}
    warnings: list[str] = []
    for cat in sorted(groups_by_cat):
        groups = groups_by_cat[cat]
        forced_train = [g for g, members in groups.items() if all(m["source_type"] in train_only for m in members)]
        eligible = [g for g in groups if g not in set(forced_train)]
        sub, warn = du.split_group_list(eligible, float(scfg.get("validation_ratio", 0.1)),
                                        float(scfg.get("test_ratio", 0.1)), (seed, "dapt-split", cat),
                                        int(scfg.get("min_eval_groups_per_category", 1)))
        assignment.update(sub)
        for g in forced_train:
            assignment[g] = "train"
        warnings += [f"{cat}: {w}" for w in warn]
        if forced_train:
            LOGGER.info("%s: %d group(s) of source types %s kept in train only", cat, len(forced_train),
                        sorted(train_only))

    splits = {s: [] for s in SPLITS}
    for d in docs:
        splits[assignment[d["group_id"]]].append(d)
    return splits, {"group_notes": dict(group_notes), "warnings": warnings}


def leakage_checks(splits: dict[str, list[dict]]) -> dict:
    """Hard checks: no group, conversation or identical text may appear in two splits."""
    result = {}
    for key in ("group_id", "text_sha1"):
        sets = {s: {d[key] for d in docs} for s, docs in splits.items()}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = sets[a] & sets[b]
            result[f"{key}:{a}&{b}"] = len(overlap)
    conv = {s: {(d["category"], d["source_name"], d["conversation_id"]) for d in docs} for s, docs in splits.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        result[f"conversation_id:{a}&{b}"] = len(conv[a] & conv[b])
    result["passed"] = all(v == 0 for k, v in result.items() if k != "passed")
    return result


def dataset_statistics(splits: dict[str, list[dict]]) -> dict:
    stats: dict = {}
    for split, docs in splits.items():
        per_cat: dict = {}
        for d in docs:
            c = per_cat.setdefault(d["category"], {"documents": 0, "groups": set(), "chars": 0,
                                                     "source_types": Counter(), "languages": Counter(),
                                                     "doc_chars": []})
            c["documents"] += 1
            c["groups"].add(d["group_id"])
            c["chars"] += d["n_chars"]
            c["source_types"][d["source_type"]] += 1
            c["languages"][d["language"]] += 1
            c["doc_chars"].append(d["n_chars"])
        stats[split] = {
            cat: {"documents": v["documents"], "groups": len(v["groups"]), "chars": v["chars"],
                  "source_types": dict(v["source_types"]), "languages": dict(v["languages"]),
                  "doc_chars_percentiles": du.percentiles(v["doc_chars"])}
            for cat, v in sorted(per_cat.items())
        }
    return stats


def fingerprint(splits: dict[str, list[dict]]) -> str:
    h = hashlib.sha1()
    for split in SPLITS:
        for d in sorted(splits[split], key=lambda x: x["doc_id"]):
            h.update(f"{split}\t{d['doc_id']}\t{d['text_sha1']}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def render_markdown(manifest: dict, stats: dict, leakage: dict, warnings: list[str]) -> str:
    lines = [f"# Prepared DAPT dataset\n\nFingerprint `{manifest['fingerprint']}`, created {manifest['created']}.\n"]
    rows = []
    for split, cats in stats.items():
        for cat, v in cats.items():
            rows.append([split, cat, v["documents"], v["groups"], v["chars"], v["source_types"],
                         v["doc_chars_percentiles"].get("p50")])
    lines.append(du.md_table(["split", "category", "documents", "groups", "chars", "source types",
                              "median chars/doc"], rows))
    lines.append(f"\n## Leakage checks: {'PASSED' if leakage['passed'] else 'FAILED'}\n")
    lines.append(du.md_table(["check", "overlapping items"], [[k, v] for k, v in leakage.items() if k != "passed"]))
    lines.append(f"\n## Exclusions\n\n{manifest['exclusions']}\n")
    if warnings:
        lines.append("## Warnings\n")
        lines += [f"- {w}" for w in warnings]
    return "\n".join(lines) + "\n"


def run(cfg: dict, use_existing_processed: bool = False) -> dict:
    processed_path = du.resolve_path(cfg, du.cfg_get(cfg, "paths.processed_dir", "data/processed")) / "documents.jsonl"
    if use_existing_processed:
        if not processed_path.is_file():
            raise FileNotFoundError(f"{processed_path} not found; run preprocess.py first or drop --use_existing_processed")
        docs = list(du.read_jsonl(processed_path))
        LOGGER.info("Loaded %d processed documents from %s", len(docs), processed_path)
    else:
        docs, _ = preprocess.run_preprocessing(cfg)

    excl_ids, excl_hashes, excl_info = load_exclusions(cfg)
    excluded = Counter()
    if excl_ids or excl_hashes:
        kept = []
        for d in docs:
            if (not d["conversation_id_generated"] and d["conversation_id"] in excl_ids) or d["text_sha1"] in excl_hashes:
                excluded[d["category"]] += 1
                continue
            kept.append(d)
        docs = kept
        LOGGER.info("Excluded %d document(s) listed in `exclusions`: %s", sum(excluded.values()), dict(excluded))

    splits, split_info = split_documents(cfg, docs)
    leakage = leakage_checks(splits)
    if not leakage["passed"]:
        raise RuntimeError(f"Leakage check failed: {leakage}. This indicates a bug or conflicting ids; "
                           "nothing was written.")
    stats = dataset_statistics(splits)
    warnings = list(split_info["warnings"])
    for cat in sorted({d["category"] for d in docs}):
        if not stats["validation"].get(cat):
            warnings.append(f"{cat}: no validation documents - it cannot be used for model selection.")
    if not docs:
        raise RuntimeError("No documents left after preprocessing/exclusions.")

    out_dir = du.resolve_path(cfg, du.cfg_get(cfg, "paths.prepared_dir", "data/prepared"))
    for split in SPLITS:
        du.write_jsonl(out_dir / f"{split}.jsonl", ({k: d.get(k) for k in DOC_FIELDS} for d in splits[split]))
    manifest = {
        "created": du.now_iso(),
        "fingerprint": fingerprint(splits),
        "counts": {s: len(v) for s, v in splits.items()},
        "categories": sorted({d["category"] for d in docs}),
        "split_config": cfg.get("split"),
        "seed": du.cfg_get(cfg, "project.seed", 42),
        "exclusions": {"excluded_documents": dict(excluded), **excl_info},
        "group_notes": split_info["group_notes"],
        "leakage_checks": leakage,
        "warnings": warnings,
        "versions": du.package_versions(("pyyaml", "numpy")),
    }
    du.write_json(out_dir / "manifest.json", manifest)
    reports_dir = du.resolve_path(cfg, du.cfg_get(cfg, "paths.reports_dir", "reports"))
    du.write_json(reports_dir / "dataset_stats.json", {"statistics": stats, "manifest": manifest})
    du.write_text(reports_dir / "dataset_stats.md", render_markdown(manifest, stats, leakage, warnings))
    for w in warnings:
        LOGGER.warning(w)
    LOGGER.info("Splits written to %s (fingerprint %s): %s", out_dir, manifest["fingerprint"], manifest["counts"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--use_existing_processed", action="store_true",
                        help="Reuse data/processed/documents.jsonl instead of re-running preprocessing.")
    args = parser.parse_args()
    cfg = du.load_config(args.config, args.overrides)
    du.setup_logging(cfg, "prepare_dataset")
    run(cfg, use_existing_processed=args.use_existing_processed)


if __name__ == "__main__":
    main()

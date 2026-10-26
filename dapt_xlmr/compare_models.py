"""
compare_models.py - does DAPT help?

Sub-commands
  mlm               Held-out MLM loss / exp(loss) / accuracy of several models on IDENTICAL masked
                    chunks (e.g. original XLM-R vs. each DAPT variant). Shows domain fit only.
  downstream-split  Fixed, conversation-level, label-stratified split of your LABELLED Hinglish
                    conversations; writes the test ids so prepare_dataset.py can exclude them from DAPT.
  downstream        Fine-tunes every model with the SAME classification recipe and seeds, evaluates on
                    the same held-out real conversations (precision, recall, F1, PR-AUC, ROC-AUC,
                    confusion matrix) and reports per-seed deltas plus a paired bootstrap interval.

The downstream recipe is a plain XLM-R sequence classifier over sliding windows with
conversation-level aggregation. It is a controlled probe of encoder quality, not your final
XLM-R + GRU multi-task detector - plug that model in with the same splits and seeds.

Run:
  python compare_models.py mlm --models original=FacebookAI/xlm-roberta-base dapt=outputs/xlm-roberta-dapt/mixed
  python compare_models.py downstream-split
  python compare_models.py downstream
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import math
import shutil
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import dapt_utils as du

LOGGER = logging.getLogger("dapt.compare")


def parse_models(items: list[str] | None, default: dict | None) -> "OrderedDict[str, str]":
    models: "OrderedDict[str, str]" = OrderedDict()
    if items:
        for item in items:
            if "=" not in item:
                raise SystemExit(f"--models entries must be name=path_or_hub_id, got {item!r}")
            name, ref = item.split("=", 1)
            models[name.strip()] = ref.strip()
    else:
        models.update(default or {})
    if len(models) < 1:
        raise SystemExit("No models given (use --models or the config's comparison/downstream models)")
    return models


# =============================================================================
# MLM comparison
# =============================================================================
def cmd_mlm(cfg: dict, args: argparse.Namespace) -> None:
    import mlm_utils as mu
    from evaluate_dapt import evaluate_model

    mu.check_library_versions()
    models = parse_models(args.models, du.cfg_get(cfg, "comparison.models"))
    tokenizers = {name: mu.load_tokenizer(cfg, ref) for name, ref in models.items()}
    fingerprints = {name: mu.tokenizer_fingerprint(t) for name, t in tokenizers.items()}
    if len(set(fingerprints.values())) > 1:
        raise SystemExit(f"Tokenizers differ between models {fingerprints}; MLM losses are not comparable.")
    names = list(models)
    bundle = mu.eval_bundle(cfg, tokenizers[names[0]], args.split, args.categories)
    results = {}
    for name, ref in models.items():
        results[name] = evaluate_model(cfg, ref, bundle=bundle, tokenizer=tokenizers[name],
                                       with_pll=True if args.pll else None)
    ref_name = names[0]
    categories = sorted(results[ref_name]["scores"], key=lambda c: (c == "ALL", c))
    rows = []
    for cat in categories:
        base = results[ref_name]["scores"][cat]
        for name in names:
            s = results[name]["scores"][cat]
            rows.append([cat, name, s["masked_tokens"], s["mlm_loss"], s["mlm_perplexity"], s["top1_accuracy"],
                         "" if name == ref_name else s["mlm_loss"] - base["mlm_loss"],
                         "" if name == ref_name else f"{(s['mlm_perplexity'] / base['mlm_perplexity'] - 1):+.1%}"])
    md = [f"# MLM comparison on `{bundle[2]['split']}`\n",
          f"Reference: `{ref_name}`. Same tokenizer, same chunks, same masked positions (seed "
          f"{du.cfg_get(cfg, 'evaluation.mask_seed', 1234)}) for every model.\n",
          du.md_table(["category", "model", "masked tokens", "MLM loss", "exp(loss)", "top-1 acc",
                       "delta loss vs ref", "exp(loss) change"], rows),
          "\nLower loss on held-out conversations means the encoder models this domain's text better. "
          "It does not show that scam detection improves - run `compare_models.py downstream` for that.\n"]
    if args.pll:
        rows = [[cat, name, v["pseudo_perplexity"]] for name in names
                for cat, v in sorted(results[name].get("pseudo_ll", {}).items())]
        md += ["## Pseudo-perplexity (sample)\n", du.md_table(["category", "model", "pseudo-perplexity"], rows)]
    out = du.resolve_path(cfg, du.cfg_get(cfg, "paths.reports_dir", "reports"))
    du.write_json(out / f"compare_mlm_{bundle[2]['split']}.json", {"models": dict(models), "results": results})
    du.write_text(out / f"compare_mlm_{bundle[2]['split']}.md", "\n".join(md) + "\n")
    LOGGER.info("\n%s", "\n".join(md[2:3]))
    LOGGER.info("Report: %s", out / f"compare_mlm_{bundle[2]['split']}.md")


# =============================================================================
# Downstream data
# =============================================================================
def load_labelled_docs(cfg: dict) -> tuple[list[dict], dict]:
    import preprocess

    dcfg = cfg.get("downstream") or {}
    if not dcfg.get("sources"):
        raise SystemExit("downstream.sources is empty; list your labelled conversation files in config.yaml")
    label_field = dcfg.get("label_field", "label")
    local_cfg = copy.deepcopy(cfg)
    local_cfg["label_fields"] = sorted(set(cfg.get("label_fields") or []) | {label_field})
    docs, report = preprocess.run_preprocessing(local_cfg, sources=dcfg["sources"], keep_labels=True,
                                                write_outputs=False)
    positives = dcfg.get("positive_labels") or du.LABEL_POSITIVE_DEFAULT
    negatives = dcfg.get("negative_labels") or du.LABEL_NEGATIVE_DEFAULT
    kept, unlabelled, unknown, conflicts = [], 0, Counter(), []
    for d in docs:
        if label_field not in d["labels"]:
            unlabelled += 1
            continue
        if label_field in d.get("label_conflicts", []):
            conflicts.append(d["doc_id"])
            continue
        y = du.normalise_binary_label(d["labels"][label_field], positives, negatives)
        if y is None:
            unknown[str(d["labels"][label_field])] += 1
            continue
        d["y"] = y
        d["raw_label"] = d["labels"][label_field]
        kept.append(d)
    if unknown:
        raise SystemExit(f"Unrecognised label values {dict(unknown)}; add them to downstream.positive_labels or "
                         "downstream.negative_labels.")
    if conflicts:
        raise SystemExit(f"{len(conflicts)} conversation(s) have conflicting labels across records, e.g. {conflicts[:3]}")
    info = {"documents": len(docs), "labelled": len(kept), "unlabelled_dropped": unlabelled,
            "removed_by_preprocessing": report["removed_by_category_and_reason"]}
    return kept, info


def cmd_downstream_split(cfg: dict, args: argparse.Namespace) -> None:
    dcfg = cfg["downstream"]
    docs, info = load_labelled_docs(cfg)
    if not docs:
        raise SystemExit("No labelled conversations found.")
    du.assign_group_ids(docs, du.cfg_get(cfg, "split.group_by", "conversation_id"),
                        du.cfg_get(cfg, "split.group_namespace", "source"))
    group_label: dict[str, int] = {}
    group_types: dict[str, set] = defaultdict(set)
    for d in docs:
        if d["group_id"] in group_label and group_label[d["group_id"]] != d["y"]:
            raise SystemExit(f"Group {d['group_id']} contains both labels; fix the data or split.group_by")
        group_label[d["group_id"]] = d["y"]
        group_types[d["group_id"]].add(d["source_type"])
    train_only = {str(t) for t in dcfg.get("train_only_source_types") or []}
    seed = int(du.cfg_get(cfg, "project.seed", 42))
    assignment: dict[str, str] = {}
    warnings = []
    for y in (0, 1):
        groups = [g for g, lab in group_label.items() if lab == y]
        forced = [g for g in groups if group_types[g] <= train_only]
        eligible = [g for g in groups if g not in set(forced)]
        sub, warn = du.split_group_list(eligible, float(dcfg.get("validation_ratio", 0.15)),
                                        float(dcfg.get("test_ratio", 0.25)), (seed, "downstream-split", y), 1)
        assignment.update(sub)
        assignment.update({g: "train" for g in forced})
        warnings += [f"label {y}: {w}" for w in warn]
    splits = {s: [d for d in docs if assignment[d["group_id"]] == s] for s in ("train", "validation", "test")}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if {d["group_id"] for d in splits[a]} & {d["group_id"] for d in splits[b]}:
            raise RuntimeError("Group leakage between downstream splits")
    out_dir = du.resolve_path(cfg, dcfg.get("split_dir", "data/downstream"))
    fields = ("doc_id", "group_id", "conversation_id", "conversation_id_generated", "source_name", "source_type",
              "language", "text", "y", "raw_label")
    for s, rows in splits.items():
        du.write_jsonl(out_dir / f"{s}.jsonl", ({k: d.get(k) for k in fields} for d in rows))
    test_ids = sorted({d["conversation_id"] for d in splits["test"] if not d["conversation_id_generated"]})
    held_ids = sorted({d["conversation_id"] for s in ("validation", "test") for d in splits[s]
                       if not d["conversation_id_generated"]})
    du.write_text(out_dir / "test_conversation_ids.txt", "\n".join(test_ids) + ("\n" if test_ids else ""))
    du.write_text(out_dir / "heldout_conversation_ids.txt", "\n".join(held_ids) + ("\n" if held_ids else ""))
    counts = {s: dict(Counter((d["y"], d["source_type"]) for d in rows)) for s, rows in splits.items()}
    counts = {s: {f"label={k[0]},{k[1]}": v for k, v in c.items()} for s, c in counts.items()}
    min_cls = int(dcfg.get("min_test_conversations_per_class", 5))
    test_pos = sum(d["y"] for d in splits["test"])
    test_neg = len(splits["test"]) - test_pos
    if test_pos < min_cls or test_neg < min_cls:
        warnings.append(f"Test set has {test_pos} scam and {test_neg} non-scam conversations (< {min_cls} per class): "
                        "metrics will be too noisy to support any claim about DAPT.")
    report = {"created": du.now_iso(), "counts": counts, "info": info, "warnings": warnings,
              "files": {s: str(out_dir / f"{s}.jsonl") for s in splits}}
    du.write_json(out_dir / "split_report.json", report)
    for w in warnings:
        LOGGER.warning(w)
    LOGGER.info("Downstream split written to %s: %s", out_dir, {s: len(v) for s, v in splits.items()})
    LOGGER.info("To keep test conversations out of DAPT, add to config.yaml:\n  exclusions:\n"
                "    conversation_id_files: [%s]\n    text_files: [%s]\nthen re-run prepare_dataset.py and train_dapt.py.",
                out_dir / "test_conversation_ids.txt", out_dir / "test.jsonl")


# =============================================================================
# Downstream fine-tuning comparison
# =============================================================================
def build_windows(docs: list[dict], tokenizer, max_len: int, overlap: int):
    import mlm_utils as mu
    from datasets import Dataset

    prefix, suffix = mu.special_template(tokenizer)
    body = max_len - len(prefix) - len(suffix)
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    owners: list[str] = []
    encoded = tokenizer([d["text"] for d in docs], add_special_tokens=False)["input_ids"] if docs else []
    for d, ids in zip(docs, encoded):
        for s, e in du.sliding_windows(len(ids), body, overlap):
            full = prefix + list(ids[s:e]) + suffix
            rows["input_ids"].append(full)
            rows["attention_mask"].append([1] * len(full))
            rows["labels"].append(int(d["y"]))
            owners.append(d["doc_id"])
    return Dataset.from_dict(rows), owners


def oversample(ds, seed: int):
    labels = list(ds["labels"])
    by = defaultdict(list)
    for i, y in enumerate(labels):
        by[y].append(i)
    if len(by) < 2:
        return ds
    target = max(len(v) for v in by.values())
    idx = []
    for y, members in sorted(by.items()):
        rng = du.rng_for(seed, "oversample", y)
        extra = [rng.choice(members) for _ in range(target - len(members))]
        idx += members + extra
    return ds.select(sorted(idx))


def aggregate(owners: list[str], probs: list[float], how: str) -> dict[str, float]:
    per: dict[str, list[float]] = defaultdict(list)
    for o, p in zip(owners, probs):
        per[o].append(p)
    if how == "max":
        return {o: max(v) for o, v in per.items()}
    return {o: sum(v) / len(v) for o, v in per.items()}


def finetune_once(cfg: dict, ref: str, tokenizer, data: dict, seed: int, out_dir: Path) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, set_seed

    import mlm_utils as mu

    dcfg = cfg["downstream"]
    set_seed(seed)
    precision = mu.resolve_precision(dcfg.get("mixed_precision", "auto"))
    model = AutoModelForSequenceClassification.from_pretrained(
        du.resolve_model_ref(cfg, ref), num_labels=2, id2label={0: "normal", 1: "scam"},
        label2id={"normal": 0, "scam": 1})
    train_ds = data["train"][0]
    if dcfg.get("class_balance", "none") == "oversample":
        train_ds = oversample(train_ds, seed)
    bs = int(dcfg.get("per_device_train_batch_size", 8))
    accum = int(dcfg.get("gradient_accumulation_steps", 1))
    epochs = float(dcfg.get("num_train_epochs", 5))
    n_dev = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    steps = max(1, math.ceil(math.ceil(len(train_ds) / (bs * n_dev)) / accum)) * math.ceil(epochs)
    args = mu.build_training_arguments(
        output_dir=str(out_dir), seed=seed, data_seed=seed, num_train_epochs=epochs,
        per_device_train_batch_size=bs, per_device_eval_batch_size=int(dcfg.get("per_device_eval_batch_size", 16)),
        gradient_accumulation_steps=accum, learning_rate=float(dcfg.get("learning_rate", 2e-5)),
        weight_decay=float(dcfg.get("weight_decay", 0.01)),
        warmup_steps=int(math.ceil(float(dcfg.get("warmup_ratio", 0.1)) * steps)),
        lr_scheduler_type="linear", optim="adamw_torch", fp16=precision["fp16"], bf16=precision["bf16"],
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, logging_strategy="steps", logging_steps=10,
        report_to="none", remove_unused_columns=False, dataloader_num_workers=0,
        # Unlike train_dapt.py, this full fine-tune (no frozen layers) had no memory-saving
        # option at all; on an 8GB GPU doing 4 models x 3 seeds back to back, this matters.
        gradient_checkpointing=bool(dcfg.get("gradient_checkpointing", False)))
    trainer = mu.make_trainer(tokenizer=tokenizer, model=model, args=args, train_dataset=train_ds,
                              eval_dataset=data["validation"][0],
                              data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
    trainer.train()
    out = {}
    for split in ("validation", "test"):
        logits = trainer.predict(data[split][0]).predictions
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = np.asarray(logits, dtype=np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        probs = (np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))[:, 1].tolist()
        out[split] = aggregate(data[split][1], probs, dcfg.get("aggregation", "mean"))
    del trainer, model
    mu.free_memory()
    if not dcfg.get("keep_checkpoints", False):
        shutil.rmtree(out_dir, ignore_errors=True)
    return out


def cmd_downstream(cfg: dict, args: argparse.Namespace) -> None:
    import mlm_utils as mu

    mu.check_library_versions()
    dcfg = cfg["downstream"]
    split_dir = du.resolve_path(cfg, dcfg.get("split_dir", "data/downstream"))
    docs = {}
    for s in ("train", "validation", "test"):
        path = split_dir / f"{s}.jsonl"
        if not path.is_file():
            raise SystemExit(f"{path} not found; run `python compare_models.py downstream-split` first.")
        docs[s] = list(du.read_jsonl(path))
        if not docs[s]:
            raise SystemExit(f"Downstream {s} split is empty; adjust downstream.validation_ratio/test_ratio or add data.")
    y_true = {s: {d["doc_id"]: int(d["y"]) for d in docs[s]} for s in ("validation", "test")}
    test_pos = sum(y_true["test"].values())
    test_neg = len(y_true["test"]) - test_pos
    min_cls = int(dcfg.get("min_test_conversations_per_class", 5))
    insufficient = test_pos < min_cls or test_neg < min_cls
    if (test_pos == 0 or test_neg == 0) and not args.allow_small:
        raise SystemExit(f"The test split has {test_pos} scam / {test_neg} non-scam conversations; both classes are "
                         "needed for PR-AUC and F1. Add labelled data (pass --allow_small to run anyway).")
    if insufficient:
        LOGGER.warning("Only %d scam / %d non-scam test conversations: results are NOT evidence either way.",
                       test_pos, test_neg)

    models = parse_models(args.models, dcfg.get("models"))
    seeds = [int(s) for s in (dcfg.get("seeds") or [42])]
    out_root = du.resolve_path(cfg, dcfg.get("output_dir", "outputs/downstream"))
    max_len = int(dcfg.get("max_seq_length", 256))
    overlap = int(dcfg.get("window_overlap", 64))
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for name, ref in models.items():
        tokenizer = mu.load_tokenizer(cfg, ref)
        data = {s: build_windows(docs[s], tokenizer, max_len, overlap) for s in ("train", "validation", "test")}
        LOGGER.info("%s: windows train=%d validation=%d test=%d", name, *(len(data[s][0]) for s in data))
        for seed in seeds:
            LOGGER.info("Fine-tuning %s (%s) with seed %d ...", name, ref, seed)
            try:
                probs = finetune_once(cfg, ref, tokenizer, data, seed, out_root / "runs" / name / f"seed{seed}")
            except Exception as exc:  # noqa: BLE001
                if mu.is_cuda_oom(exc):
                    mu.free_memory()
                    raise SystemExit("CUDA out of memory during downstream fine-tuning: lower "
                                     "downstream.per_device_train_batch_size (and raise gradient_accumulation_steps) "
                                     "or downstream.max_seq_length.") from exc
                raise
            ids_val = sorted(y_true["validation"])
            ids_test = sorted(y_true["test"])
            yv = [y_true["validation"][i] for i in ids_val]
            yt = [y_true["test"][i] for i in ids_test]
            pv = [probs["validation"][i] for i in ids_val]
            pt = [probs["test"][i] for i in ids_test]
            tuned = du.best_f1_threshold(yv, pv)
            runs[name][seed] = {"test_probs": dict(zip(ids_test, pt)),
                                "at_0.5": du.binary_metrics(yt, pt, 0.5),
                                "at_val_tuned": du.binary_metrics(yt, pt, tuned)}
            out_root.mkdir(parents=True, exist_ok=True)
            with (out_root / f"predictions_{name}_seed{seed}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["doc_id", "y_true", "p_scam"])
                w.writerows([[i, y, p] for i, y, p in zip(ids_test, yt, pt)])
            m = runs[name][seed]["at_0.5"]
            LOGGER.info("  seed %d: F1 %.3f | P %.3f | R %.3f | PR-AUC %.3f", seed, m["f1"], m["precision"],
                        m["recall"], m["pr_auc"])

    ids_test = sorted(y_true["test"])
    yt = [y_true["test"][i] for i in ids_test]
    summary, averaged = {}, {}
    for name in models:
        agg = {}
        for view in ("at_0.5", "at_val_tuned"):
            for metric in ("precision", "recall", "f1", "pr_auc", "roc_auc"):
                agg[f"{view}:{metric}"] = du.mean_std([runs[name][s][view][metric] for s in seeds])
        averaged[name] = [sum(runs[name][s]["test_probs"][i] for s in seeds) / len(seeds) for i in ids_test]
        agg["seed_averaged_probs@0.5"] = du.binary_metrics(yt, averaged[name], 0.5)
        summary[name] = agg
    ref_name = list(models)[0]
    comparisons = {}
    for name in list(models)[1:]:
        per_seed = {s: {m: runs[name][s]["at_0.5"][m] - runs[ref_name][s]["at_0.5"][m] for m in ("f1", "pr_auc")}
                    for s in seeds}
        comparisons[f"{name} - {ref_name}"] = {
            "per_seed_delta": per_seed,
            "bootstrap_pr_auc": du.paired_bootstrap(yt, averaged[ref_name], averaged[name], "pr_auc", 0.5,
                                                    int(dcfg.get("bootstrap_samples", 1000)), 0),
            "bootstrap_f1@0.5": du.paired_bootstrap(yt, averaged[ref_name], averaged[name], "f1", 0.5,
                                                    int(dcfg.get("bootstrap_samples", 1000)), 1),
        }
    result = {"created": du.now_iso(), "models": dict(models), "seeds": seeds, "test_positives": test_pos,
              "test_negatives": test_neg, "insufficient_data": insufficient, "summary": summary,
              "comparisons": comparisons,
              "per_run": {n: {s: {k: v for k, v in r.items() if k != "test_probs"} for s, r in rs.items()}
                          for n, rs in runs.items()}}
    reports = du.resolve_path(cfg, du.cfg_get(cfg, "paths.reports_dir", "reports")) / "downstream"
    du.write_json(reports / "downstream_comparison.json", result)
    du.write_text(reports / "downstream_comparison.md", render_downstream(result))
    LOGGER.info("Report: %s", reports / "downstream_comparison.md")


def render_downstream(r: dict) -> str:
    lines = ["# Downstream comparison (conversation-level, held-out real test set)\n"]
    if r["insufficient_data"]:
        lines.append(f"> **INSUFFICIENT DATA** - {r['test_positives']} scam / {r['test_negatives']} non-scam test "
                     "conversations. These numbers cannot support a claim that DAPT helps or hurts.\n")
    lines.append(f"Seeds: {r['seeds']}. Same splits, windows, hyper-parameters and seeds for every model.\n")
    rows = []
    for name, agg in r["summary"].items():
        def ms(key):
            mu_, sd = agg[key]
            return "n/a" if math.isnan(mu_) else f"{mu_:.3f} ± {sd:.3f}"
        rows.append([name, ms("at_0.5:precision"), ms("at_0.5:recall"), ms("at_0.5:f1"), ms("at_0.5:pr_auc"),
                     ms("at_0.5:roc_auc"), ms("at_val_tuned:f1")])
    lines.append(du.md_table(["model", "precision@0.5", "recall@0.5", "F1@0.5", "PR-AUC", "ROC-AUC",
                              "F1@val-tuned threshold"], rows))
    lines.append("\n## Confusion matrices (seed-averaged probabilities, threshold 0.5)\n")
    rows = [[n, a["seed_averaged_probs@0.5"]["confusion_matrix"]] for n, a in r["summary"].items()]
    lines.append(du.md_table(["model", "tn / fp / fn / tp"], rows))
    if r["comparisons"]:
        lines.append("\n## Paired comparison\n")
        for key, c in r["comparisons"].items():
            lines.append(f"### {key}\n")
            lines.append(du.md_table(["seed", "delta F1@0.5", "delta PR-AUC"],
                                     [[s, d["f1"], d["pr_auc"]] for s, d in c["per_seed_delta"].items()]))
            for b in ("bootstrap_pr_auc", "bootstrap_f1@0.5"):
                bs = c[b]
                if bs.get("valid_resamples"):
                    lines.append(f"\n{b}: mean delta {bs['mean_delta']:+.3f}, 95% CI [{bs['ci95'][0]:+.3f}, "
                                 f"{bs['ci95'][1]:+.3f}], P(delta>0) = {bs['p_delta_gt_0']:.2f} "
                                 f"({bs['valid_resamples']} resamples)")
            lines.append("\nA confidence interval that includes 0 means no demonstrated difference.\n")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("mlm", help="held-out MLM comparison")
    m.add_argument("--models", nargs="*", default=None, help="name=path_or_hub_id ... (first = reference)")
    m.add_argument("--split", default="test", choices=["validation", "test"])
    m.add_argument("--categories", nargs="*", default=None)
    m.add_argument("--pll", action="store_true")
    sub.add_parser("downstream-split", help="fixed labelled split + exclusion lists")
    d = sub.add_parser("downstream", help="supervised fine-tuning comparison")
    d.add_argument("--models", nargs="*", default=None)
    d.add_argument("--allow_small", action="store_true", help="run even if a test class is empty")
    args = p.parse_args()
    cfg = du.load_config(args.config, args.overrides)
    du.setup_logging(cfg, f"compare_models_{args.command.replace('-', '_')}")
    {"mlm": cmd_mlm, "downstream-split": cmd_downstream_split, "downstream": cmd_downstream}[args.command](cfg, args)


if __name__ == "__main__":
    main()

"""
train_classifier.py - trains the XLM-R + GRU multi-task scam detector (methodology SS4.6):
call-level scam probability + per-turn manipulation-tactic multi-label classification.

Not part of the original dapt-xlmr toolkit. Built on top of it: the encoder starts from a
DAPT checkpoint (see config.yaml `classifier.encoder_path`), everything else (GRU, two
heads) is new and trained from scratch here, on the same fixed conversation split the
downstream encoder comparison used (data/downstream/{train,validation,test}.jsonl), so the
classifier's results sit on splits already known to be leakage-free.

Run:
  python train_classifier.py --dry_run        # build the data, print shapes, train nothing
  python train_classifier.py --smoke_test     # ~1 min correctness check, tiny subset, 1 epoch
  python train_classifier.py                  # full run: classifier.seeds, classifier.epochs
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import shutil
from pathlib import Path

import dapt_utils as du

LOGGER = logging.getLogger("dapt.classifier")


def build_optimizer(model, encoder_lr: float, head_lr: float, weight_decay: float):
    import torch

    encoder_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.AdamW(
        [{"params": encoder_params, "lr": encoder_lr}, {"params": head_params, "lr": head_lr}],
        weight_decay=weight_decay)


def move_batch(batch: dict, device) -> dict:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def evaluate(model, loader, device, autocast_dtype, n_tactics: int):
    import torch

    model.eval()
    scam_probs: dict[str, float] = {}
    scam_true: dict[str, int] = {}
    tactic_probs: list = []
    tactic_true: list = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with du_autocast(device, autocast_dtype):
                out = model(batch["input_ids"], batch["attention_mask"], batch["n_turns"])
            probs = torch.sigmoid(out["scam_logit"]).float().cpu().tolist()
            for cid, p, y in zip(batch["conversation_id"], probs, batch["y"].cpu().tolist()):
                scam_probs[cid] = p
                scam_true[cid] = int(y)
            tac_p = torch.sigmoid(out["tactic_logits"]).float().cpu()
            mask = batch["turn_mask"].cpu().bool()
            tac_t = batch["tactic_targets"].cpu()
            for b in range(tac_p.shape[0]):
                m = mask[b]
                tactic_probs.append(tac_p[b][m])
                tactic_true.append(tac_t[b][m])
    tactic_probs_cat = torch.cat(tactic_probs, dim=0) if tactic_probs else torch.zeros(0, n_tactics)
    tactic_true_cat = torch.cat(tactic_true, dim=0) if tactic_true else torch.zeros(0, n_tactics)
    return scam_probs, scam_true, tactic_probs_cat, tactic_true_cat


def du_autocast(device, dtype):
    import mlm_utils as mu
    return mu.autocast_ctx(device, dtype)


def tactic_metrics(probs, true, tactic_labels: list[str], threshold: float = 0.5) -> dict:
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    if probs.shape[0] == 0:
        nan = float("nan")
        return {"macro_f1": nan, "per_label": {name: {"precision": nan, "recall": nan, "f1": nan}
                                                for name in tactic_labels}}
    pred = (probs.numpy() >= threshold).astype(int)
    truth = true.numpy().astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(truth, pred, average=None, zero_division=0)
    macro = f1_score(truth, pred, average="macro", zero_division=0)
    return {"macro_f1": float(macro),
            "per_label": {name: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
                          for i, name in enumerate(tactic_labels)}}


def run_training(cfg: dict, tokenizer, datasets: dict, seed: int, out_dir: Path, smoke_test: bool) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup, set_seed

    import classifier_data as cd
    import classifier_model as cm
    import mlm_utils as mu

    ccfg = cfg["classifier"]
    tactic_labels = ccfg["tactic_labels"]
    n_tactics = len(tactic_labels)
    set_seed(seed)
    device = mu.eval_device()
    precision = mu.resolve_precision(ccfg.get("mixed_precision", "auto"))

    encoder_path = du.resolve_model_ref(cfg, ccfg["encoder_path"])
    model = cm.XLMRGRUMultiTask(encoder_path, n_tactics, gru_hidden=int(ccfg["gru_hidden"]),
                                dropout=float(ccfg["dropout"]), bidirectional=bool(ccfg["bidirectional"]))
    model.set_gradient_checkpointing(bool(ccfg.get("gradient_checkpointing", False)))
    model.to(device)

    bs = int(ccfg["batch_size"])
    accum = int(ccfg["gradient_accumulation_steps"])
    epochs = 1 if smoke_test else int(ccfg["epochs"])
    train_loader = DataLoader(datasets["train"], batch_size=bs, shuffle=True, collate_fn=cd.collate)
    val_loader = DataLoader(datasets["validation"], batch_size=bs, shuffle=False, collate_fn=cd.collate)
    test_loader = DataLoader(datasets["test"], batch_size=bs, shuffle=False, collate_fn=cd.collate)

    optimizer = build_optimizer(model, float(ccfg["encoder_lr"]), float(ccfg["head_lr"]), float(ccfg["weight_decay"]))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(float(ccfg["warmup_ratio"]) * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0
    best_state = None
    patience = int(ccfg["early_stopping_patience"])
    patience_left = patience
    tactic_weight = float(ccfg["tactic_loss_weight"])

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        running = {"loss": 0.0, "scam_loss": 0.0, "tactic_loss": 0.0}
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            with du_autocast(device, precision["autocast_dtype"]):
                out = model(batch["input_ids"], batch["attention_mask"], batch["n_turns"])
                losses = cm.compute_loss(out, batch, tactic_weight)
            (losses["loss"] / accum).backward()
            running["loss"] += float(losses["loss"].detach())
            running["scam_loss"] += float(losses["scam_loss"])
            running["tactic_loss"] += float(losses["tactic_loss"])
            if (step + 1) % accum == 0 or step == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        n = len(train_loader)
        LOGGER.info("seed %d epoch %d/%d: loss=%.4f scam_loss=%.4f tactic_loss=%.4f", seed, epoch + 1, epochs,
                   running["loss"] / n, running["scam_loss"] / n, running["tactic_loss"] / n)

        mu.free_memory()  # 2026-10-22: a 7.5GB GPU hit a fragmentation OOM between epochs
        # (reserved-but-unallocated exceeded the failed allocation) despite plenty of total
        # free memory; defragmenting the CUDA cache here fixed it.
        sp, st, tp, tt = evaluate(model, val_loader, device, precision["autocast_dtype"], n_tactics)
        mu.free_memory()
        ids = sorted(sp)
        y_true = [st[i] for i in ids]
        probs = [sp[i] for i in ids]
        scam_val = du.binary_metrics(y_true, probs, 0.5)
        tac_val = tactic_metrics(tp, tt, tactic_labels)
        combined = (scam_val["f1"] + tac_val["macro_f1"]) / 2
        LOGGER.info("seed %d epoch %d/%d: val scam_f1=%.3f val_tactic_macro_f1=%.3f combined=%.3f", seed,
                   epoch + 1, epochs, scam_val["f1"], tac_val["macro_f1"], combined)
        if combined > best_metric:
            best_metric = combined
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                LOGGER.info("seed %d: early stopping after epoch %d (no improvement for %d epochs)", seed,
                           epoch + 1, patience)
                break

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best_model.pt")

    sp_val, st_val, _, _ = evaluate(model, val_loader, device, precision["autocast_dtype"], n_tactics)
    ids_val = sorted(sp_val)
    tuned_threshold = du.best_f1_threshold([st_val[i] for i in ids_val], [sp_val[i] for i in ids_val])

    sp_test, st_test, tp_test, tt_test = evaluate(model, test_loader, device, precision["autocast_dtype"], n_tactics)
    ids_test = sorted(sp_test)
    y_test = [st_test[i] for i in ids_test]
    p_test = [sp_test[i] for i in ids_test]
    result = {
        "seed": seed,
        "scam_at_0.5": du.binary_metrics(y_test, p_test, 0.5),
        "scam_at_val_tuned": du.binary_metrics(y_test, p_test, tuned_threshold),
        "tactic": tactic_metrics(tp_test, tt_test, tactic_labels),
        "test_probs": dict(zip(ids_test, p_test)),
        "test_true": dict(zip(ids_test, y_test)),
    }
    del model
    mu.free_memory()
    if not ccfg.get("keep_checkpoints", False):
        shutil.rmtree(out_dir, ignore_errors=True)
    return result


def render_report(cfg: dict, results: list[dict]) -> str:
    ccfg = cfg["classifier"]
    tactic_labels = ccfg["tactic_labels"]
    lines = ["# XLM-R + GRU multi-task scam detector - held-out test results\n",
            f"Encoder: `{ccfg['encoder_path']}`. Seeds: {[r['seed'] for r in results]}. "
            "Scam head: sigmoid probability reported as a percentage. Tactic head: per-turn "
            "multi-label, same threshold (0.5) for all seeds.\n"]
    f1 = [r["scam_at_0.5"]["f1"] for r in results]
    pr = [r["scam_at_0.5"]["pr_auc"] for r in results]
    roc = [r["scam_at_0.5"]["roc_auc"] for r in results]
    prec = [r["scam_at_0.5"]["precision"] for r in results]
    rec = [r["scam_at_0.5"]["recall"] for r in results]
    f1_tuned = [r["scam_at_val_tuned"]["f1"] for r in results]
    macro = [r["tactic"]["macro_f1"] for r in results]

    def fmt(vals):
        mu_, sd = du.mean_std(vals)
        return f"{mu_:.3f} +/- {sd:.3f}"

    lines.append("## Scam detection (call level)\n")
    lines.append(du.md_table(
        ["precision@0.5", "recall@0.5", "F1@0.5", "PR-AUC", "ROC-AUC", "F1@val-tuned threshold"],
        [[fmt(prec), fmt(rec), fmt(f1), fmt(pr), fmt(roc), fmt(f1_tuned)]]))
    lines.append("\n## Manipulation-tactic detection (per turn, multi-label)\n")
    lines.append(f"Macro-F1 across seeds: {fmt(macro)}\n")
    rows = []
    for name in tactic_labels:
        p = [r["tactic"]["per_label"][name]["precision"] for r in results]
        rr = [r["tactic"]["per_label"][name]["recall"] for r in results]
        ff = [r["tactic"]["per_label"][name]["f1"] for r in results]
        rows.append([name, fmt(p), fmt(rr), fmt(ff)])
    lines.append(du.md_table(["tactic", "precision", "recall", "F1"], rows))
    lines.append("\nThis is the actual scam-detection deliverable, not a probe like "
                "`compare_models.py downstream` - its numbers measure real detection performance, "
                "not just which encoder reads this domain's text better.\n")
    return "\n".join(lines) + "\n"


def cmd_train(cfg: dict, args: argparse.Namespace) -> None:
    import csv

    import mlm_utils as mu

    mu.check_library_versions()
    ccfg = cfg["classifier"]
    tokenizer = mu.load_tokenizer(cfg, ccfg["encoder_path"])

    import classifier_data as cd

    datasets, info = cd.build_datasets(cfg, tokenizer)
    LOGGER.info("conversations: train=%d validation=%d test=%d", info["train"], info["validation"], info["test"])

    if args.dry_run:
        example = datasets["train"][0]
        LOGGER.info("example tensor shapes: input_ids=%s attention_mask=%s tactic_targets=%s n_turns=%d",
                   tuple(example["input_ids"].shape), tuple(example["attention_mask"].shape),
                   tuple(example["tactic_targets"].shape), example["n_turns"])
        steps_per_epoch = math.ceil(len(datasets["train"]) / int(ccfg["batch_size"]) /
                                    int(ccfg["gradient_accumulation_steps"]))
        LOGGER.info("~%d optimizer steps/epoch x %d epochs, %d seed(s)", steps_per_epoch, int(ccfg["epochs"]),
                   len(ccfg["seeds"]))
        return

    out_root = du.resolve_path(cfg, ccfg.get("output_dir", "outputs/classifier"))
    seeds = [int(ccfg["seeds"][0])] if args.smoke_test else [int(s) for s in ccfg["seeds"]]
    if args.smoke_test:
        for split in datasets:
            datasets[split].examples = datasets[split].examples[:8]
        LOGGER.info("--smoke_test: 1 seed, 1 epoch, 8 conversations per split")

    results = []
    for seed in seeds:
        LOGGER.info("=== seed %d ===", seed)
        result = run_training(cfg, tokenizer, datasets, seed, out_root / "runs" / f"seed{seed}", args.smoke_test)
        results.append(result)
        LOGGER.info("seed %d test: scam F1@0.5=%.3f tactic macro-F1=%.3f", seed, result["scam_at_0.5"]["f1"],
                   result["tactic"]["macro_f1"])

    out_root.mkdir(parents=True, exist_ok=True)
    for r in results:
        path = out_root / f"predictions_seed{r['seed']}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["conversation_id", "y_true", "p_scam"])
            for cid in sorted(r["test_probs"]):
                w.writerow([cid, r["test_true"][cid], r["test_probs"][cid]])

    report = render_report(cfg, results)
    reports_dir = du.resolve_path(cfg, du.cfg_get(cfg, "paths.reports_dir", "reports")) / "classifier"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "classifier_report.md"
    du.write_text(report_path, report)
    LOGGER.info("Report: %s", report_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dry_run", action="store_true", help="build data, print shapes/step counts, train nothing")
    p.add_argument("--smoke_test", action="store_true", help="1 seed, 1 epoch, 8 conversations per split")
    args = p.parse_args()
    cfg = du.load_config(args.config, args.overrides)
    du.setup_logging(cfg, "train_classifier")
    cmd_train(cfg, args)


if __name__ == "__main__":
    main()

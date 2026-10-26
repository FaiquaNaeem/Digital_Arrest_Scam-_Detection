"""
train_dapt.py - continued masked-language-model pretraining (DAPT) of XLM-RoBERTa.

Data flow
  data/prepared/{train,validation}.jsonl   (from prepare_dataset.py; already split by conversation)
    -> tokenised with the ORIGINAL XLM-R tokenizer and cut into chunks per conversation (cached)
    -> training chunks re-sampled per category (sampling.* in config) for each pass
    -> validation chunks masked once with a fixed seed (stable validation loss)
    -> Trainer: dynamic masking (DataCollatorForLanguageModeling), AdamW, warmup + linear decay,
       mixed precision, periodic evaluation + checkpoints, early stopping, best-checkpoint reload
    -> final model + tokenizer saved in Hugging Face format, then re-loaded to verify

Examples
  python train_dapt.py --smoke_test                    # a few steps on a small subset first
  python train_dapt.py --experiment mixed               # full run
  python train_dapt.py --experiment mixed --resume_from_checkpoint auto
  python train_dapt.py --experiment hinglish_only --dry_run   # data mix + step plan only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path

import dapt_utils as du
import mlm_utils as mu

import torch  # noqa: E402  (after mlm_utils sets TOKENIZERS_PARALLELISM)

LOGGER = logging.getLogger("dapt.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--experiment", default=None, help="korean_only | hinglish_only | mixed | any key of `experiments`")
    p.add_argument("--run_name", default=None, help="folder under outputs/runs/ (default: experiment name)")
    p.add_argument("--smoke_test", action="store_true", help="tiny subset, few steps, separate run folder")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="'auto' = latest checkpoint of this run (or start fresh if none), or a checkpoint path")
    p.add_argument("--force_resume", action="store_true",
                   help="resume even if data/sampling/optimiser settings differ from the interrupted run")
    p.add_argument("--dry_run", action="store_true", help="build data, mixture report and step plan; no training")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return p.parse_args()


def apply_smoke_overrides(cfg: dict) -> tuple[int, int]:
    s = cfg.get("smoke_test") or {}
    cfg.setdefault("tokenization", {})["max_seq_length"] = int(s.get("max_seq_length", 128))
    t = cfg.setdefault("training", {})
    t["max_steps"] = int(s.get("max_steps", 10))
    t["eval_steps"] = int(s.get("eval_steps", 5))
    t["logging_steps"] = 1
    t["save_total_limit"] = 1
    return int(s.get("max_train_chunks", 64)), int(s.get("max_eval_chunks", 32))


def _sha1_ints(values: list[int]) -> str:
    return hashlib.sha1(",".join(map(str, values)).encode()).hexdigest()[:16]


def mixture_markdown(report: dict, eval_info: dict) -> str:
    rows = [[c, r["natural_tokens"], f"{r['natural_share']:.1%}", f"{r['target_share']:.1%}",
             f"{r['effective_share']:.1%}", f"{r['upsample_factor']:.2f}x", f"{r['distinct_chunk_coverage']:.0%}"]
            for c, r in report["categories"].items()]
    text = ["# Training data mixture\n",
            f"Strategy `{report['strategy']}`" + (f" (alpha={report['alpha']})" if report["alpha"] is not None else "")
            + f", {report['num_passes']} pass(es), {report['total_training_chunks']:,} training chunks, "
              f"{report['total_training_tokens']:,} tokens (excluding special tokens).\n",
            du.md_table(["category", "natural tokens", "natural share", "target share", "effective share",
                         "repeat factor / pass", "chunks used"], rows),
            "\n## Validation set used for checkpoint selection\n",
            du.md_table(["category", "chunks", "tokens"], [[c, v["chunks"], v["tokens"]] for c, v in eval_info.items()])]
    if report["warnings"]:
        text.append("\n## Warnings\n")
        text += [f"- {w}" for w in report["warnings"]]
    return "\n".join(text) + "\n"


class GuardedRun:
    """Run a step; on CUDA OOM write a report with concrete fixes and exit with code 3.
    Every other exception propagates unchanged."""

    def __init__(self, run_dir: Path, cfg: dict, resume_cmd: str):
        self.run_dir, self.cfg, self.resume_cmd = run_dir, cfg, resume_cmd

    def __call__(self, fn, what: str):
        try:
            return fn()
        except KeyboardInterrupt:
            LOGGER.warning("Interrupted during %s. Resume later with:\n  %s", what, self.resume_cmd)
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is an OOM
            if not mu.is_cuda_oom(exc):
                raise
            t = self.cfg["training"]
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
            mu.free_memory()
            report = {
                "when": what, "time": du.now_iso(), "error": str(exc).splitlines()[0][:500],
                "per_device_train_batch_size": t.get("per_device_train_batch_size"),
                "per_device_eval_batch_size": t.get("per_device_eval_batch_size"),
                "gradient_accumulation_steps": t.get("gradient_accumulation_steps"),
                "max_seq_length": du.cfg_get(self.cfg, "tokenization.max_seq_length"),
                "gradient_checkpointing": t.get("gradient_checkpointing"),
                "peak_memory_gb": peak,
                "suggestions": [
                    "Halve training.per_device_train_batch_size and double training.gradient_accumulation_steps "
                    "(same effective batch, so you can still resume).",
                    "Set training.gradient_checkpointing=true (about 25-30% slower, much less activation memory).",
                    "If it failed during evaluation, lower training.per_device_eval_batch_size.",
                    "Lower tokenization.max_seq_length (e.g. 128); this re-chunks the data, so start a new run.",
                    "Restart the Colab runtime to clear fragmented memory; export "
                    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before launching.",
                ],
                "resume_with": self.resume_cmd,
            }
            du.write_json(self.run_dir / "oom_report.json", report)
            LOGGER.error("CUDA out of memory during %s (peak %.2f GB). Nothing was hidden: details in %s",
                         what, peak or float("nan"), self.run_dir / "oom_report.json")
            for s in report["suggestions"]:
                LOGGER.error("  - %s", s)
            raise SystemExit(3) from exc


def main() -> None:
    args = parse_args()
    cfg = du.load_config(args.config, args.overrides)
    experiment = args.experiment or du.cfg_get(cfg, "project.experiment")
    if experiment not in cfg["experiments"]:
        raise SystemExit(f"Unknown experiment {experiment!r}; choose from {sorted(cfg['experiments'])}")
    smoke_limits = apply_smoke_overrides(cfg) if args.smoke_test else None
    tcfg = cfg["training"]
    run_name = args.run_name or tcfg.get("run_name") or experiment
    if args.smoke_test and not run_name.endswith("-smoke"):
        run_name += "-smoke"
    run_dir = du.resolve_path(cfg, du.cfg_get(cfg, "paths.output_root", "outputs")) / "runs" / run_name
    ckpt_dir = run_dir / "checkpoints"
    du.setup_logging(cfg, "train_dapt", log_dir=run_dir / "logs")
    versions = mu.check_library_versions()

    from transformers import EarlyStoppingCallback, TrainerCallback, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    seed = int(du.cfg_get(cfg, "project.seed", 42))
    set_seed(seed)

    # ---------------------------------------------------------------- resume
    existing = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.is_dir() else None
    resume_path = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            resume_path = existing
            if resume_path:
                LOGGER.info("Resuming from %s", resume_path)
            else:
                LOGGER.info("No checkpoint in %s yet; starting from the pretrained model.", ckpt_dir)
        else:
            resume_path = str(Path(args.resume_from_checkpoint).expanduser().resolve())
            if not Path(resume_path).is_dir():
                raise SystemExit(f"Checkpoint folder not found: {resume_path}")
    elif existing and args.smoke_test and not args.dry_run:
        LOGGER.info("Clearing checkpoints of the previous smoke test in %s", ckpt_dir)
        shutil.rmtree(ckpt_dir)
    elif existing and not args.dry_run:
        raise SystemExit(f"{ckpt_dir} already contains {Path(existing).name}. Continue it with "
                         "--resume_from_checkpoint auto, or use a new --run_name.")
    resume_cmd = (f"python train_dapt.py --config {args.config} --experiment {experiment} --run_name {run_name}"
                  + (" --smoke_test" if args.smoke_test else "") + " --resume_from_checkpoint auto"
                  + "".join(f" --set {o}" for o in args.overrides))

    # ---------------------------------------------------------------- data
    tokenizer = mu.load_tokenizer(cfg)
    msl = int(du.cfg_get(cfg, "tokenization.max_seq_length", 256))
    if tokenizer.model_max_length < 100_000 and msl > tokenizer.model_max_length:
        raise SystemExit(f"max_seq_length={msl} exceeds the tokenizer limit {tokenizer.model_max_length}")
    splits, cache_info = mu.tokenized_splits(cfg, tokenizer, msl, splits=("train", "validation"))
    train_full, val_full = splits["train"], splits["validation"]
    if train_full is None:
        raise SystemExit("The training split is empty.")
    categories = list(cfg["experiments"][experiment]["categories"])
    by_cat = mu.chunks_by_category(train_full, categories)
    missing = [c for c in categories if c not in by_cat]
    if missing:
        raise SystemExit(f"Experiment {experiment!r}: no training chunks for {missing}. Check sources and split.")

    scfg = cfg.get("sampling") or {}
    epochs_cfg = float(tcfg.get("num_train_epochs", 1))
    passes = max(1, math.ceil(epochs_cfg))
    if passes != epochs_cfg:
        LOGGER.warning("num_train_epochs=%s rounded up to %d re-sampled passes", epochs_cfg, passes)
    indices, mix_report = du.sample_mixture(
        by_cat, scfg.get("strategy", "temperature"), float(scfg.get("alpha", 0.5)), scfg.get("custom_weights"),
        scfg.get("total_train_tokens"), float(scfg.get("max_upsample_factor") or 0), passes, seed)
    if smoke_limits:
        du.rng_for(seed, "smoke-train").shuffle(indices)
        indices = indices[:smoke_limits[0]]
    n_tokens_col = list(train_full["n_tokens"])
    prefix, suffix = mu.special_template(tokenizer)
    mean_tokens = sum(n_tokens_col[i] for i in indices) / max(1, len(indices)) + len(prefix) + len(suffix)
    train_ds = train_full.select(indices).select_columns(["input_ids", "attention_mask", "special_tokens_mask"])

    if val_full is None:
        raise SystemExit("The validation split is empty; it is needed for evaluation and early stopping.")
    sel_cats = list(tcfg.get("selection_categories") or categories)
    val_by_cat = mu.chunks_by_category(val_full, sel_cats)
    lacking = [c for c in sel_cats if c not in val_by_cat]
    if lacking:
        LOGGER.warning("No validation chunks for %s; they are not part of checkpoint selection.", lacking)
    if not val_by_cat:
        raise SystemExit(f"No validation chunks for selection categories {sel_cats}")
    if tcfg.get("eval_mix", "balanced") == "balanced":
        val_idx = du.balanced_subset(val_by_cat, seed)
    else:
        val_idx = sorted(i for items in val_by_cat.values() for i, _ in items)
    if smoke_limits:
        du.rng_for(seed, "smoke-eval").shuffle(val_idx)
        val_idx = sorted(val_idx[:smoke_limits[1]])
    val_cats = list(val_full["category"])
    val_ntok = list(val_full["n_tokens"])
    eval_info: dict = {}
    for i in val_idx:
        e = eval_info.setdefault(val_cats[i], {"chunks": 0, "tokens": 0})
        e["chunks"] += 1
        e["tokens"] += int(val_ntok[i])
    mlm_prob = float(tcfg.get("mlm_probability", 0.15))
    eval_ds = mu.fixed_masked_dataset(val_full.select(val_idx), tokenizer, mlm_prob,
                                      int(du.cfg_get(cfg, "evaluation.mask_seed", 1234)))
    eval_ds = eval_ds.select_columns(["input_ids", "attention_mask", "labels"])

    # ---------------------------------------------------------------- plan
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world = int(os.environ.get("WORLD_SIZE", "1"))
    per_dev = int(tcfg.get("per_device_train_batch_size", 4))
    accum = int(tcfg.get("gradient_accumulation_steps", 1))
    micro_global = per_dev * (world if world > 1 else max(1, n_gpu))
    effective_batch = micro_global * accum
    steps_per_pass_total = max(1, math.ceil(math.ceil(len(train_ds) / micro_global) / accum))
    max_steps_cfg = int(tcfg.get("max_steps", -1) or -1)
    total_steps = max_steps_cfg if max_steps_cfg > 0 else steps_per_pass_total
    warmup_steps = int(math.ceil(float(tcfg.get("warmup_ratio", 0.06)) * total_steps))
    eval_steps = int(tcfg.get("eval_steps", 200))
    if eval_steps > total_steps:
        new = max(1, total_steps // 4)
        LOGGER.warning("eval_steps=%d > total steps %d; evaluating every %d steps instead", eval_steps, total_steps, new)
        eval_steps = new
    logging_steps = max(1, min(int(tcfg.get("logging_steps", 25)), eval_steps))
    plan = {"train_chunks": len(train_ds), "distinct_train_chunks": len(set(indices)), "eval_chunks": len(eval_ds),
            "mean_tokens_per_chunk": round(mean_tokens, 1), "per_device_train_batch_size": per_dev,
            "gradient_accumulation_steps": accum, "devices": max(1, n_gpu), "effective_batch_size": effective_batch,
            "total_optimizer_steps": total_steps, "warmup_steps": warmup_steps, "eval_and_save_every": eval_steps,
            "logging_every": logging_steps, "max_seq_length": msl}
    for k, v in plan.items():
        LOGGER.info("plan | %-28s %s", k, v)

    identity = json.loads(json.dumps({
        "experiment": experiment, "categories": categories, "model": du.cfg_get(cfg, "model.name_or_path"),
        "revision": du.cfg_get(cfg, "model.revision"), "data_fingerprint": cache_info["data_fingerprint"],
        "tokenizer_fingerprint": cache_info["tokenizer_fingerprint"], "max_seq_length": msl,
        "sampling_strategy": scfg.get("strategy", "temperature"), "passes": passes,
        "train_indices_sha1": _sha1_ints(indices), "eval_indices_sha1": _sha1_ints(val_idx),
        "mlm_probability": mlm_prob, "effective_batch_size": effective_batch,
        "learning_rate": float(tcfg.get("learning_rate")), "seed": seed, "smoke_test": bool(args.smoke_test),
    }, default=str))
    manifest_path = run_dir / "run_manifest.json"
    if resume_path and manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8")).get("identity", {})
        diffs = {k: {"before": old.get(k), "now": v} for k, v in identity.items() if old.get(k) != v}
        if diffs and not args.force_resume:
            raise SystemExit("Cannot resume: these settings differ from the interrupted run:\n"
                             + json.dumps(diffs, indent=2, ensure_ascii=False)
                             + "\nRestore them, start a new --run_name, or pass --force_resume.")
        if diffs:
            LOGGER.warning("Resuming despite changed settings (--force_resume): %s", list(diffs))
    run_dir.mkdir(parents=True, exist_ok=True)
    du.write_json(manifest_path, {"identity": identity, "plan": plan, "created": du.now_iso(), "versions": versions,
                                  "resumed_from": resume_path, "cache": cache_info, "config": du.config_snapshot(cfg)})
    du.write_json(run_dir / "data_mixture.json", {"mixture": mix_report, "selection_eval": eval_info})
    du.write_text(run_dir / "data_mixture.md", mixture_markdown(mix_report, eval_info))
    for c, r in mix_report["categories"].items():
        LOGGER.info("mix  | %-15s natural %5.1f%% -> effective %5.1f%% (x%.2f per pass)", c,
                    100 * r["natural_share"], 100 * r["effective_share"], r["upsample_factor"])
    for w in mix_report["warnings"]:
        LOGGER.warning(w)
    if args.dry_run:
        LOGGER.info("Dry run finished. Mixture report: %s", run_dir / "data_mixture.md")
        return

    # ---------------------------------------------------------------- model + trainer
    precision = mu.resolve_precision(tcfg.get("mixed_precision", "auto"))
    LOGGER.info("Precision: %s", precision["label"])
    model = mu.load_mlm_model(cfg, du.cfg_get(cfg, "model.name_or_path"))
    n_emb = model.get_input_embeddings().num_embeddings
    if n_emb != len(tokenizer):
        LOGGER.warning("Embedding rows (%d) != tokenizer size (%d); the tokenizer is not the model's own.",
                       n_emb, len(tokenizer))
    pos_limit = getattr(model.config, "max_position_embeddings", 514)
    if getattr(model.config, "model_type", "") in ("xlm-roberta", "roberta"):
        pos_limit -= 2  # RoBERTa-style position ids start after the padding index
    if msl > pos_limit:
        raise SystemExit(f"max_seq_length={msl} exceeds the model's position limit {pos_limit}")

    gc_on = bool(tcfg.get("gradient_checkpointing", False))
    report_to = tcfg.get("report_to", "none")
    if isinstance(report_to, list) and not report_to:
        report_to = "none"
    training_args = mu.build_training_arguments(
        output_dir=str(ckpt_dir), run_name=run_name, seed=seed, data_seed=seed,
        num_train_epochs=1, max_steps=total_steps if max_steps_cfg > 0 else -1,
        per_device_train_batch_size=per_dev,
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=accum,
        learning_rate=float(tcfg.get("learning_rate", 5e-5)), weight_decay=float(tcfg.get("weight_decay", 0.01)),
        adam_beta1=float(tcfg.get("adam_beta1", 0.9)), adam_beta2=float(tcfg.get("adam_beta2", 0.98)),
        adam_epsilon=float(tcfg.get("adam_epsilon", 1e-6)), max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "linear"), warmup_steps=warmup_steps, optim="adamw_torch",
        fp16=precision["fp16"], bf16=precision["bf16"],
        gradient_checkpointing=gc_on, gradient_checkpointing_kwargs={"use_reentrant": False} if gc_on else None,
        eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps,
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        logging_strategy="steps", logging_steps=logging_steps, logging_first_step=True,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, remove_unused_columns=False,
        dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=torch.cuda.is_available(),
        group_by_length=bool(tcfg.get("group_by_length", True)),
        report_to=report_to, full_determinism=bool(tcfg.get("full_determinism", False)),
        auto_find_batch_size=bool(tcfg.get("auto_find_batch_size", False)),
    )

    class TrainingMonitor(TrainerCallback):
        """Writes every log record to metrics.jsonl and adds throughput and GPU memory."""

        def __init__(self, path: Path):
            self.path = path
            self.last_t = None
            self.last_step = 0
            self.eval_time = 0.0

        def on_train_begin(self, args, state, control, **kwargs):
            self.last_t, self.last_step = time.time(), state.global_step
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            self.eval_time += float((metrics or {}).get("eval_runtime", 0.0) or 0.0)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero:
                return
            logs = dict(logs or {})
            rec = {"time": du.now_iso(), "step": state.global_step, "epoch": state.epoch}
            if "loss" in logs and self.last_t is not None:
                now = time.time()
                dt = max(1e-6, now - self.last_t - self.eval_time)
                dsteps = state.global_step - self.last_step
                if dsteps > 0:
                    rec["optimizer_steps_per_sec"] = dsteps / dt
                    rec["train_samples_per_sec"] = dsteps * effective_batch / dt
                    rec["est_train_tokens_per_sec"] = rec["train_samples_per_sec"] * mean_tokens
                self.last_t, self.last_step, self.eval_time = now, state.global_step, 0.0
                if torch.cuda.is_available():
                    rec["gpu_peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            rec.update(logs)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            if "loss" in logs:
                LOGGER.info("step %6d | loss %.4f | lr %.2e | %.1f samples/s | ~%.0f tokens/s%s", state.global_step,
                            logs["loss"], logs.get("learning_rate", float("nan")),
                            rec.get("train_samples_per_sec", float("nan")), rec.get("est_train_tokens_per_sec", float("nan")),
                            f" | peak {rec['gpu_peak_memory_gb']} GB" if "gpu_peak_memory_gb" in rec else "")
            elif "eval_loss" in logs:
                LOGGER.info("step %6d | eval_loss %.4f | exp(eval_loss) %.2f", state.global_step, logs["eval_loss"],
                            du.safe_exp(logs["eval_loss"]))

    callbacks = [TrainingMonitor(run_dir / "metrics.jsonl")]
    patience = int(tcfg.get("early_stopping_patience", 3) or 0)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience,
                                               early_stopping_threshold=float(tcfg.get("early_stopping_threshold", 0.0))))
    else:
        LOGGER.info("Early stopping disabled: training runs the full budget; the best checkpoint is still reloaded.")
    trainer = mu.make_trainer(tokenizer=tokenizer, model=model, args=training_args, train_dataset=train_ds,
                              eval_dataset=eval_ds, data_collator=mu.MLMCollator(tokenizer, mlm_prob, 8),
                              callbacks=callbacks)
    guarded = GuardedRun(run_dir, cfg, resume_cmd)

    baseline = None
    if not resume_path and tcfg.get("evaluate_before_training", True):
        LOGGER.info("Evaluating the ORIGINAL model on the selection set (baseline) ...")
        baseline = guarded(trainer.evaluate, "baseline evaluation")
        du.write_json(run_dir / "baseline_eval.json", baseline)

    # ---------------------------------------------------------------- train
    started = time.time()
    result = guarded(lambda: trainer.train(resume_from_checkpoint=resume_path), "training")
    trainer.save_state()
    train_metrics = dict(result.metrics)
    du.write_json(run_dir / "train_metrics.json", train_metrics)
    final_eval = guarded(trainer.evaluate, "final evaluation")
    du.write_json(run_dir / "final_eval.json", final_eval)
    LOGGER.info("Training finished in %.1f min. Best checkpoint: %s (eval_loss %s)", (time.time() - started) / 60,
                trainer.state.best_model_checkpoint, trainer.state.best_metric)

    # ---------------------------------------------------------------- save + verify
    if args.smoke_test:
        final_dir = run_dir / "final-model"
    else:
        final_dir = du.resolve_path(cfg, du.format_template(
            tcfg.get("final_model_dir", "outputs/xlm-roberta-dapt/{experiment}"), experiment=experiment,
            run_name=run_name))
    if args.smoke_test and final_dir.exists():
        shutil.rmtree(final_dir)
    elif final_dir.exists() and any(final_dir.iterdir()):
        backup = final_dir.with_name(f"{final_dir.name}.previous-{_dt.datetime.now():%Y%m%d-%H%M%S}")
        final_dir.rename(backup)
        LOGGER.warning("Existing model folder moved to %s", backup)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    info = {
        "base_model": du.cfg_get(cfg, "model.name_or_path"), "revision": du.cfg_get(cfg, "model.revision"),
        "experiment": experiment, "categories": categories, "run_name": run_name, "smoke_test": bool(args.smoke_test),
        "global_step": trainer.state.global_step, "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint, "baseline_eval": baseline,
        "final_eval": final_eval, "train_metrics": train_metrics, "data_fingerprint": cache_info["data_fingerprint"],
        "tokenizer_fingerprint": cache_info["tokenizer_fingerprint"], "mixture": mix_report, "plan": plan,
        "versions": versions, "created": du.now_iso(),
    }
    du.write_json(final_dir / "dapt_info.json", info)
    du.write_text(final_dir / "README.md", mu.model_card(info))

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok2 = AutoTokenizer.from_pretrained(str(final_dir))
    model2 = AutoModelForMaskedLM.from_pretrained(str(final_dir))
    if mu.tokenizer_fingerprint(tok2) != cache_info["tokenizer_fingerprint"]:
        raise RuntimeError("Saved tokenizer differs from the training tokenizer")
    same = torch.equal(model2.get_input_embeddings().weight.detach().float().cpu(),
                       trainer.model.get_input_embeddings().weight.detach().float().cpu())
    rows = eval_ds[: min(2, len(eval_ds))]
    batch = mu.pad_premasked(rows["input_ids"], rows["attention_mask"], rows["labels"], tok2.pad_token_id)
    with torch.no_grad():
        loss = float(model2(**batch).loss)
    if not (same and math.isfinite(loss)):
        raise RuntimeError(f"Reload check failed (weights equal={same}, loss={loss})")
    LOGGER.info("Saved and verified: %s (AutoTokenizer + AutoModelForMaskedLM reload OK, sample loss %.3f)",
                final_dir, loss)

    summary = [f"# DAPT run `{run_name}`\n",
               f"- experiment: `{experiment}` ({', '.join(categories)})",
               f"- steps: {trainer.state.global_step} / planned {total_steps}",
               f"- baseline eval_loss (original model): {baseline.get('eval_loss') if baseline else 'not run'}",
               f"- best eval_loss: {trainer.state.best_metric}",
               f"- final model: `{final_dir}`",
               "", mixture_markdown(mix_report, eval_info)]
    du.write_text(run_dir / "summary.md", "\n".join(summary))
    LOGGER.info("Run summary: %s", run_dir / "summary.md")


if __name__ == "__main__":
    main()

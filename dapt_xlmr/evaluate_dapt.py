"""
evaluate_dapt.py - held-out masked-language-model evaluation of one model.

What it measures (per category and overall, on chunks of held-out conversations):
  * mlm_loss        token-weighted cross-entropy over masked positions
  * mlm_perplexity  exp(mlm_loss). NOT a true perplexity: an MLM does not define a normalised
                    probability of a whole sequence. It is a monotone transform of the masked-token
                    loss and is only comparable between models that share the tokenizer, the masking
                    rate and the masked positions - all three are fixed here (same tokenizer check,
                    deterministic masks seeded by chunk id).
  * top1_accuracy   share of masked tokens predicted exactly
  * optional pseudo-perplexity (--pll): every token masked in turn (Salazar et al., 2020); a
    well-defined sequence score for MLMs, but one forward pass per token, so it runs on a sample.

None of this says anything about scam-detection accuracy; see compare_models.py downstream.

Run:  python evaluate_dapt.py --model outputs/xlm-roberta-dapt/mixed --split test
"""
from __future__ import annotations

import argparse
import logging
import re

import dapt_utils as du
import mlm_utils as mu

LOGGER = logging.getLogger("dapt.evaluate")


def model_tag(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(ref).strip("/"))[-80:]


def evaluate_model(cfg: dict, model_ref: str, split: str = "test", categories: list[str] | None = None,
                   with_pll: bool | None = None, bundle=None, tokenizer=None) -> dict:
    """Evaluate one MLM. `bundle` = (chunk_ds, masked_ds, info) can be shared between models."""
    tokenizer = tokenizer or mu.load_tokenizer(cfg, model_ref)
    if bundle is None:
        bundle = mu.eval_bundle(cfg, tokenizer, split, categories)
    chunk_ds, masked_ds, info = bundle
    if mu.tokenizer_fingerprint(tokenizer) != info["tokenizer_fingerprint"]:
        raise RuntimeError(f"{model_ref} uses a different tokenizer than the evaluation data; losses would not be "
                           "comparable.")
    precision = mu.resolve_precision(du.cfg_get(cfg, "training.mixed_precision", "auto"))
    LOGGER.info("Loading %s ...", model_ref)
    model = mu.load_mlm_model(cfg, model_ref)
    scores = mu.masked_lm_scores(model, masked_ds, tokenizer.pad_token_id,
                                 int(du.cfg_get(cfg, "evaluation.batch_size", 16)), precision["autocast_dtype"])
    result = {"model": model_ref, "split": info["split"], "max_seq_length": info["max_seq_length"],
              "data_fingerprint": info["data_fingerprint"], "mask_seed": du.cfg_get(cfg, "evaluation.mask_seed", 1234),
              "mlm_probability": du.cfg_get(cfg, "evaluation.mlm_probability", 0.15), "scores": scores}
    pll_cfg = du.cfg_get(cfg, "evaluation.pll", {}) or {}
    run_pll = pll_cfg.get("enabled", False) if with_pll is None else with_pll
    if run_pll:
        LOGGER.info("Computing pseudo-log-likelihood on up to %s sequences per category ...",
                    pll_cfg.get("max_sequences_per_category", 20))
        result["pseudo_ll"] = mu.pseudo_log_likelihood(
            model, chunk_ds, tokenizer, int(pll_cfg.get("max_sequences_per_category", 20)),
            int(pll_cfg.get("max_seq_length", 128)), int(pll_cfg.get("batch_size", 64)),
            int(du.cfg_get(cfg, "project.seed", 42)), precision["autocast_dtype"])
    del model
    mu.free_memory()
    return result


def render_markdown(result: dict) -> str:
    rows = [[k, v["chunks"], v["masked_tokens"], v["mlm_loss"], v["mlm_perplexity"], v["top1_accuracy"]]
            for k, v in sorted(result["scores"].items(), key=lambda kv: (kv[0] == "ALL", kv[0]))]
    lines = [f"# MLM evaluation: `{result['model']}`\n",
             f"Split `{result['split']}`, max_seq_length {result['max_seq_length']}, mask seed {result['mask_seed']}, "
             f"masking rate {result['mlm_probability']}, data `{result['data_fingerprint']}`.\n",
             du.md_table(["category", "chunks", "masked tokens", "MLM loss", "exp(loss)", "top-1 acc"], rows),
             "\nexp(loss) is comparable only between models with the same tokenizer on these same masked positions.\n"]
    if "pseudo_ll" in result:
        rows = [[k, v["sequences"], v["tokens"], v["pll_per_token"], v["pseudo_perplexity"]]
                for k, v in sorted(result["pseudo_ll"].items(), key=lambda kv: (kv[0] == "ALL", kv[0]))]
        lines += ["## Pseudo-log-likelihood\n", du.md_table(["category", "sequences", "tokens", "PLL/token",
                                                               "pseudo-perplexity"], rows)]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model", default=None, help="model folder or hub id (default: final model of --experiment)")
    p.add_argument("--experiment", default=None)
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--pll", action="store_true", help="also compute pseudo-perplexity on a sample")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    cfg = du.load_config(args.config, args.overrides)
    du.setup_logging(cfg, "evaluate_dapt")
    mu.check_library_versions()
    experiment = args.experiment or du.cfg_get(cfg, "project.experiment")
    model_ref = args.model or str(du.resolve_path(cfg, du.format_template(
        du.cfg_get(cfg, "training.final_model_dir", "outputs/xlm-roberta-dapt/{experiment}"),
        experiment=experiment, run_name=experiment)))
    result = evaluate_model(cfg, model_ref, args.split, args.categories, with_pll=True if args.pll else None)
    out_dir = du.resolve_path(cfg, du.cfg_get(cfg, "paths.reports_dir", "reports")) / "mlm_eval"
    stem = f"{model_tag(model_ref)}_{result['split']}"
    du.write_json(out_dir / f"{stem}.json", result)
    du.write_text(out_dir / f"{stem}.md", render_markdown(result))
    for cat, s in sorted(result["scores"].items()):
        LOGGER.info("%-16s loss %.4f | exp %.2f | acc %.3f | %d masked tokens", cat, s["mlm_loss"],
                    s["mlm_perplexity"], s["top1_accuracy"], s["masked_tokens"])
    LOGGER.info("Report: %s", out_dir / f"{stem}.md")


if __name__ == "__main__":
    main()

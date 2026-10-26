"""
risk_score.py - the methodology's final stage: turn the trained XLM-R + GRU classifier's
output into one 0-100 risk score + a Low/Medium/High band for a call.

No fixed risk-score formula was specified for this project, so this is a documented,
tunable default (see config.yaml `risk_scoring` and PROJECT_LOG.md), not a derived result:

    risk_score = 100 * clip(
        weights.scam_probability * scam_probability
      + weights.tactic_presence  * tactic_presence
      + weights.escalation       * escalation
    , 0, 1)

  - scam_probability: the classifier's own call-level scam probability (dominant signal).
  - tactic_presence:  severity-weighted average, over the six tactics, of each tactic's
                      peak per-turn probability anywhere in the call ("did this tactic show
                      up at all, and how clearly").
  - escalation:       how much more intense the (severity-weighted) tactic signal is in the
                      second half of the call than the first half, clipped to >= 0. Digital-
                      arrest calls characteristically escalate pressure as they go on; a
                      flat or front-loaded tactic profile is less distinctively "arrest-call"
                      shaped even at the same average intensity.

Run:
  python risk_score.py --test              # score the held-out test set, write a report
  python risk_score.py --conversation call.json   # score one new conversation (see --help)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import dapt_utils as du

LOGGER = logging.getLogger("dapt.risk_score")


def load_scorer(cfg: dict):
    import torch

    import classifier_model as cm
    import mlm_utils as mu

    ccfg = cfg["classifier"]
    rcfg = cfg["risk_scoring"]
    tokenizer = mu.load_tokenizer(cfg, ccfg["encoder_path"])
    encoder_path = du.resolve_model_ref(cfg, ccfg["encoder_path"])
    model = cm.XLMRGRUMultiTask(encoder_path, len(ccfg["tactic_labels"]), gru_hidden=int(ccfg["gru_hidden"]),
                                dropout=float(ccfg["dropout"]), bidirectional=bool(ccfg["bidirectional"]))
    ckpt_path = du.resolve_path(cfg, rcfg["checkpoint"])
    if not ckpt_path.is_file():
        raise SystemExit(f"{ckpt_path} not found; train_classifier.py must be run with "
                         "classifier.keep_checkpoints=true for at least one seed first.")
    device = mu.eval_device()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    precision = mu.resolve_precision(ccfg.get("mixed_precision", "auto"))
    return model, tokenizer, device, precision


def run_inference(model, tokenizer, device, precision, cleaned_texts: list[str], max_seq_length: int):
    """One conversation -> (scam_probability: float, tactic_probs: list[list[float]] per turn)."""
    import torch

    import mlm_utils as mu

    enc = tokenizer(cleaned_texts, padding="max_length", truncation=True, max_length=max_seq_length,
                    return_tensors="pt")
    input_ids = enc["input_ids"].unsqueeze(0).to(device)
    attention_mask = enc["attention_mask"].unsqueeze(0).to(device)
    n_turns = torch.tensor([len(cleaned_texts)], device=device)
    with torch.no_grad():
        with mu.autocast_ctx(device, precision["autocast_dtype"]):
            out = model(input_ids, attention_mask, n_turns)
    scam_prob = float(torch.sigmoid(out["scam_logit"])[0].cpu())
    tactic_probs = torch.sigmoid(out["tactic_logits"])[0].cpu().tolist()  # (n_turns, n_tactics)
    return scam_prob, tactic_probs


def tactic_presence_and_escalation(tactic_probs: list[list[float]], tactic_labels: list[str],
                                   severity: dict[str, float]) -> tuple[float, float, dict[str, float]]:
    """severity-weighted peak-per-tactic presence, and escalation between call halves."""
    n = len(tactic_probs)
    weights = [float(severity.get(name, 0.5)) for name in tactic_labels]
    wsum = sum(weights) or 1.0
    peak = [max(turn[i] for turn in tactic_probs) for i in range(len(tactic_labels))]
    presence = sum(w * p for w, p in zip(weights, peak)) / wsum
    per_tactic_peak = dict(zip(tactic_labels, peak))

    if n < 2:
        return presence, 0.0, per_tactic_peak
    mid = n // 2
    first, second = tactic_probs[:mid], tactic_probs[mid:]

    def weighted_mean(turns: list[list[float]]) -> float:
        if not turns:
            return 0.0
        return sum(w * sum(turn[i] for turn in turns) / len(turns) for i, w in enumerate(weights)) / wsum

    escalation = max(0.0, weighted_mean(second) - weighted_mean(first))
    return presence, escalation, per_tactic_peak


def compute_risk_score(cfg: dict, scam_prob: float, tactic_probs: list[list[float]]) -> dict:
    rcfg = cfg["risk_scoring"]
    tactic_labels = cfg["classifier"]["tactic_labels"]
    presence, escalation, per_tactic_peak = tactic_presence_and_escalation(
        tactic_probs, tactic_labels, rcfg["tactic_severity"])
    w = rcfg["weights"]
    raw = (float(w["scam_probability"]) * scam_prob + float(w["tactic_presence"]) * presence +
          float(w["escalation"]) * escalation)
    score = round(100 * max(0.0, min(1.0, raw)), 1)
    bands = rcfg["bands"]
    band = "Low" if score < bands["low_max"] else ("Medium" if score < bands["medium_max"] else "High")
    return {"risk_score": score, "band": band, "scam_probability": round(scam_prob, 4),
            "tactic_presence": round(presence, 4), "escalation": round(escalation, 4),
            "tactic_peak": {k: round(v, 4) for k, v in per_tactic_peak.items()}}


def score_conversation(cfg: dict, model, tokenizer, device, precision, turns_raw: list[dict]) -> dict:
    """turns_raw: list of {"speaker": str, "text": str}, caller/victim in time order."""
    import preprocess

    import classifier_data as cd

    cleaner = preprocess.TextCleaner(cfg.get("preprocessing") or {})
    cleaned_texts = cd.clean_and_tag_turns(cleaner, turns_raw)
    max_turns = int(cfg["classifier"]["max_turns"])
    if len(cleaned_texts) > max_turns:
        LOGGER.warning("conversation has %d turns, truncating to max_turns=%d", len(cleaned_texts), max_turns)
        cleaned_texts = cleaned_texts[:max_turns]
    scam_prob, tactic_probs = run_inference(model, tokenizer, device, precision, cleaned_texts,
                                            int(cfg["classifier"]["max_seq_length"]))
    return compute_risk_score(cfg, scam_prob, tactic_probs)


def cmd_test(cfg: dict) -> None:
    import csv

    import classifier_data as cd
    import mlm_utils as mu

    mu.check_library_versions()
    model, tokenizer, device, precision = load_scorer(cfg)
    assignment = cd.load_split_assignment(cfg)
    conversations = cd.load_conversations(cfg)
    test_convos = [c for c in conversations if assignment.get(c["conversation_id"]) == "test"]
    LOGGER.info("scoring %d held-out test conversations", len(test_convos))

    rows = []
    for conv in test_convos:
        texts = [t["text"] for t in conv["turns"][:int(cfg["classifier"]["max_turns"])]]
        scam_prob, tactic_probs = run_inference(model, tokenizer, device, precision, texts,
                                                int(cfg["classifier"]["max_seq_length"]))
        result = compute_risk_score(cfg, scam_prob, tactic_probs)
        result["conversation_id"] = conv["conversation_id"]
        result["y_true"] = conv["y"]
        rows.append(result)

    out_dir = du.resolve_path(cfg, cfg["risk_scoring"].get("output_dir", "outputs/risk_scoring"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "test_risk_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["conversation_id", "y_true", "risk_score", "band", "scam_probability", "tactic_presence",
                   "escalation"])
        for r in rows:
            w.writerow([r["conversation_id"], r["y_true"], r["risk_score"], r["band"], r["scam_probability"],
                       r["tactic_presence"], r["escalation"]])
    LOGGER.info("Predictions: %s", csv_path)

    report = render_report(rows)
    reports_dir = du.resolve_path(cfg, du.cfg_get(cfg, "paths.reports_dir", "reports")) / "risk_score"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "risk_score_report.md"
    du.write_text(report_path, report)
    LOGGER.info("Report: %s", report_path)


def render_report(rows: list[dict]) -> str:
    from collections import Counter

    by_label = {0: [r["risk_score"] for r in rows if r["y_true"] == 0],
               1: [r["risk_score"] for r in rows if r["y_true"] == 1]}
    lines = ["# Risk score - held-out test set\n",
            f"{len(rows)} conversations (" + ", ".join(f"{len(v)} label={k}" for k, v in by_label.items()) +
            ").\n", "## Score distribution by true label\n"]
    stat_rows = []
    for label, name in ((0, "not-scam"), (1, "scam")):
        vals = sorted(by_label[label])
        if not vals:
            continue
        n = len(vals)
        stat_rows.append([name, f"{min(vals):.1f}", f"{vals[n // 4]:.1f}", f"{vals[n // 2]:.1f}",
                          f"{vals[3 * n // 4]:.1f}", f"{max(vals):.1f}"])
    lines.append(du.md_table(["true label", "min", "p25", "median", "p75", "max"], stat_rows))
    lines.append("\n## Band assignment vs. true label\n")
    counts = Counter((r["y_true"], r["band"]) for r in rows)
    band_rows = []
    for label, name in ((0, "not-scam"), (1, "scam")):
        band_rows.append([name, counts[(label, "Low")], counts[(label, "Medium")], counts[(label, "High")]])
    lines.append(du.md_table(["true label", "Low", "Medium", "High"], band_rows))
    lines.append("\nA well-separated score puts almost all `not-scam` calls in Low and almost all `scam` "
                "calls in High; anything landing in Medium is a call this score is genuinely unsure about, "
                "which is the band's intended meaning, not a modelling failure.\n")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--test", action="store_true", help="score the held-out test set, write a report")
    p.add_argument("--conversation", metavar="FILE.json",
                   help='score one conversation: a JSON file with {"turns": [{"speaker": "caller", '
                        '"text": "..."}, {"speaker": "victim", "text": "..."}, ...]}')
    args = p.parse_args()
    cfg = du.load_config(args.config, args.overrides)
    du.setup_logging(cfg, "risk_score")
    if args.test:
        cmd_test(cfg)
    elif args.conversation:
        import mlm_utils as mu

        mu.check_library_versions()
        model, tokenizer, device, precision = load_scorer(cfg)
        turns_raw = json.loads(Path(args.conversation).read_text(encoding="utf-8"))["turns"]
        result = score_conversation(cfg, model, tokenizer, device, precision, turns_raw)
        print(json.dumps(result, indent=2))
    else:
        raise SystemExit("Pass --test or --conversation FILE.json")


if __name__ == "__main__":
    main()

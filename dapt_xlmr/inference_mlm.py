"""
inference_mlm.py - masked-token prediction with the adapted model (optionally side by side
with the original XLM-R), to eyeball what DAPT changed.

Write <mask> (or [MASK]) where a token should be predicted. Each mask position is predicted
independently (an MLM does not jointly fill several masks). Inputs get the same Unicode and
whitespace normalisation as the training data; nothing is translated.

Examples
  python inference_mlm.py --model outputs/xlm-roberta-dapt/mixed \
      --text "Aapke naam par ek <mask> case register hua hai." \
      --compare_with FacebookAI/xlm-roberta-base
  python inference_mlm.py --model outputs/xlm-roberta-dapt/mixed --file probes.txt --top_k 10
  python inference_mlm.py --model outputs/xlm-roberta-dapt/mixed --interactive
"""
from __future__ import annotations

import argparse
import logging
import re

import dapt_utils as du
import mlm_utils as mu
from preprocess import TextCleaner

import torch  # noqa: E402

LOGGER = logging.getLogger("dapt.inference")
DEMO_TEXTS = [
    "Main CBI <mask> se baat kar raha hoon, aapke Aadhaar par case darj hai.",
    "Aap yeh baat kisi ko <mask> mat batana, warna aapko arrest kar liya jayega.",
    "검찰청 <mask> 수사관입니다. 본인 명의 계좌가 범죄에 연루되었습니다.",
    "지금 바로 안전 계좌로 <mask> 하셔야 합니다.",
]


def load(cfg: dict, ref: str):
    tok = mu.load_tokenizer(cfg, ref)
    model = mu.load_mlm_model(cfg, ref)
    model.to(mu.eval_device()).eval()
    return tok, model


@torch.no_grad()
def predict(tokenizer, model, text: str, top_k: int) -> list[list[tuple[str, float]]]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    positions = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    logits = model(**enc).logits[0, positions].float()
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, k=top_k, dim=-1)
    out = []
    for row_p, row_i in zip(top.values.tolist(), top.indices.tolist()):
        tokens = tokenizer.convert_ids_to_tokens(row_i)
        out.append([(t.replace("\u2581", "_"), p) for t, p in zip(tokens, row_p)])
    return out


def prepare_text(raw: str, tokenizer, cleaner: TextCleaner) -> str:
    text = re.sub(r"\[MASK\]|<mask>", "\x00", raw, flags=re.IGNORECASE)
    text = cleaner.basic_normalize(text).replace("\x00", tokenizer.mask_token)
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model", default=None, help="adapted model folder (default: final model of --experiment)")
    p.add_argument("--experiment", default=None)
    p.add_argument("--compare_with", default=None, help="e.g. FacebookAI/xlm-roberta-base")
    p.add_argument("--text", action="append", default=[], help="text containing <mask>; repeatable")
    p.add_argument("--file", default=None, help="UTF-8 file with one masked text per line")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    cfg = du.load_config(args.config, args.overrides)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    experiment = args.experiment or du.cfg_get(cfg, "project.experiment")
    model_ref = args.model or str(du.resolve_path(cfg, du.format_template(
        du.cfg_get(cfg, "training.final_model_dir", "outputs/xlm-roberta-dapt/{experiment}"),
        experiment=experiment, run_name=experiment)))
    cleaner = TextCleaner(cfg.get("preprocessing") or {})
    models = {"adapted": load(cfg, model_ref)}
    if args.compare_with:
        models["original"] = load(cfg, args.compare_with)

    texts = list(args.text)
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            texts += [line.strip() for line in fh if line.strip()]
    if not texts and not args.interactive:
        LOGGER.info("No --text given; using built-in demo probes.")
        texts = DEMO_TEXTS

    def run(raw: str) -> None:
        print("\n" + "=" * 100 + f"\n{raw}")
        results = {}
        for name, (tok, model) in models.items():
            text = prepare_text(raw, tok, cleaner)
            if tok.mask_token not in text:
                print("  (no <mask> in this text - add one, e.g. 'Aapka <mask> block ho jayega')")
                return
            results[name] = predict(tok, model, text, args.top_k)
        n_masks = len(next(iter(results.values())))
        for m in range(n_masks):
            print(f"  mask #{m + 1}")
            header = "    " + " | ".join(f"{name:<34}" for name in results)
            print(header)
            for rank in range(args.top_k):
                cells = [f"{tok_:<24} {prob:7.2%}  " for tok_, prob in (results[n][m][rank] for n in results)]
                print(f"    {rank + 1}. " + " | ".join(cells))

    for t in texts:
        run(t)
    if args.interactive:
        print("\nInteractive mode - type a sentence with <mask>, or an empty line to quit.")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            run(line)


if __name__ == "__main__":
    main()

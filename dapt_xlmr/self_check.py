"""
self_check.py - end-to-end verification of the whole pipeline BEFORE you use real data.

It writes small throwaway fixture files (clearly artificial word salad in Korean and romanised
Hinglish, plus fake identifiers to test redaction) into a temporary folder, builds a tiny
random-weight XLM-R (2 layers, hidden size 32, the real XLM-R tokenizer), and runs every script
through its command-line interface:

  inspect -> preprocess -> prepare -> downstream-split -> prepare with exclusions
  -> train --dry_run for all experiments -> train --smoke_test
  -> interrupted training (killed after its first checkpoint) -> resume -> refusal checks
  -> evaluate -> compare mlm -> inference -> downstream fine-tuning comparison

The fixtures are never written into data/ and are deleted afterwards (use --keep to inspect).
Nothing here measures model quality; it only proves the code paths work together on your
installed library versions.

  python self_check.py               # full check (downloads the XLM-R tokenizer + config, ~5 MB)
  python self_check.py --data_only   # no torch/transformers needed
  python self_check.py --real_model  # use the real xlm-roberta-base weights (slow on CPU)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
KO_WORDS = ["안녕하세요", "고객님", "확인", "계좌", "오늘", "내일", "회의", "점심", "날씨", "커피", "주말", "택배",
            "학교", "병원", "버스", "시간", "전화", "문자", "가족", "친구", "영화", "공원", "시장", "은행"]
HI_WORDS = ["aap", "kya", "hai", "nahi", "abhi", "kal", "ghar", "office", "bhai", "theek", "accha", "paisa",
            "phone", "message", "doctor", "school", "khana", "chai", "dost", "train", "bazaar", "kaam", "baat", "yaar"]
LABEL_POOL = ["scam", "normal"]


def words(rng: random.Random, pool: list[str], n: int) -> str:
    return " ".join(rng.choice(pool) for _ in range(n))


def make_fixtures(root: Path) -> tuple[list[dict], list[dict]]:
    rng = random.Random(7)
    raw = root / "raw"
    # Korean scam: JSONL, conversation-level + turn-level records, plus deliberate defects.
    ko_scam = raw / "korean" / "scam"
    ko_scam.mkdir(parents=True)
    lines = []
    for i in range(14):
        text = "\n".join(f"[00:00:{10 + j:02d}] {words(rng, KO_WORDS, 9)} (잡음)" for j in range(6))
        if i % 3 == 0:
            text += "\n연락처는 010-1234-5678 입니다. 주민번호 900101-1234567"
        lines.append(json.dumps({"conversation_id": f"ko_scam_{i:02d}", "text": text, "label": "scam"},
                                ensure_ascii=False))
    for k in range(4):
        turns = [{"conversation_id": f"ko_scam_turns_{k}", "turn_index": j, "speaker": "AB"[j % 2],
                  "text": f"순서 {j} {words(rng, KO_WORDS, 8)}"} for j in range(5)]
        rng.shuffle(turns)  # out of order on purpose; turn_index must restore it
        lines += [json.dumps(t, ensure_ascii=False) for t in turns]
    lines.append('{"conversation_id": "broken", "text": "unterminated')     # malformed JSON line
    lines.append(json.dumps({"conversation_id": "empty_one", "text": "   "}))  # empty text
    lines.append(json.dumps({"conversation_id": "no_text", "body": "x"}))     # missing field
    dup = json.loads(lines[0])
    dup["conversation_id"] = "ko_scam_duplicate"
    lines.append(json.dumps(dup, ensure_ascii=False))                         # exact duplicate text
    (ko_scam / "calls.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Korean normal: CSV with a differently named text column, plus TXT files.
    ko_norm = raw / "korean" / "normal"
    ko_norm.mkdir(parents=True)
    rows = ["conversation_id,transcript,label"]
    for i in range(12):
        body = "\n".join(words(rng, KO_WORDS, 10) for _ in range(5)).replace('"', "")
        rows.append(f'ko_norm_{i:02d},"{body}",normal')
    (ko_norm / "daily.csv").write_text("\n".join(rows) + "\n", encoding="utf-8-sig")
    ko_txt = raw / "korean" / "normal_txt"
    ko_txt.mkdir(parents=True)
    for i in range(4):
        (ko_txt / f"chat_{i}.txt").write_text("\n".join(words(rng, KO_WORDS, 11) for _ in range(6)), encoding="utf-8")

    # Hinglish real: JSONL with labels and fake identifiers.
    hi_real = raw / "hinglish" / "real"
    hi_real.mkdir(parents=True)
    lines = []
    for i in range(16):
        label = LABEL_POOL[i % 2]
        text = "\n".join(f"Caller: {words(rng, HI_WORDS, 10)}" for _ in range(6))
        if i % 4 == 0:
            text += ("\nCaller: mera number +91 98765 43210 hai, Aadhaar 2345 6789 0123, email ramesh.k@example.com, "
                     "UPI ramesh@okaxis, PAN ABCDE1234F, amount Rs 50,000 abhi bhejo")
        lines.append(json.dumps({"conversation_id": f"hi_real_{i:02d}", "text": text, "label": label}))
    (hi_real / "real.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Hinglish synthetic: JSON list with role-named turns (roles must be neutralised).
    hi_syn = raw / "hinglish" / "synthetic"
    hi_syn.mkdir(parents=True)
    records = []
    for i in range(10):
        records.append({"conversation_id": f"hi_syn_{i:02d}", "source_type": "synthetic", "label": LABEL_POOL[i % 2],
                        "turns": [{"speaker": "scammer" if j % 2 == 0 else "victim", "text": words(rng, HI_WORDS, 9)}
                                  for j in range(6)]})
    (hi_syn / "synthetic.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    sources = [
        {"name": "korean_scam", "category": "korean_scam", "paths": [str(ko_scam)], "language": "ko",
         "source_type": "real", "text_field": "text", "conversation_id_field": "conversation_id",
         "speaker_field": "speaker", "turn_order_field": "turn_index"},
        {"name": "korean_normal_csv", "category": "korean_normal", "paths": [str(ko_norm)], "language": "ko",
         "source_type": "real", "text_field": "transcript", "conversation_id_field": "conversation_id"},
        {"name": "korean_normal_txt", "category": "korean_normal", "paths": [str(ko_txt / "*.txt")],
         "language": "ko", "source_type": "real", "txt_mode": "file"},
        {"name": "hinglish_real", "category": "hinglish", "paths": [str(hi_real)], "language": "hi-en",
         "source_type": "real", "text_field": "text", "conversation_id_field": "conversation_id"},
        {"name": "hinglish_synthetic", "category": "hinglish", "paths": [str(hi_syn)], "language": "hi-en",
         "source_type": "synthetic", "turns_field": "turns", "conversation_id_field": "conversation_id"},
        {"name": "optional_missing", "category": "hinglish", "paths": [str(raw / "does_not_exist")],
         "required": False},
    ]
    downstream = [
        {"name": "hinglish_labelled_real", "category": "hinglish_labelled", "paths": [str(hi_real)],
         "language": "hi-en", "source_type": "real", "text_field": "text", "conversation_id_field": "conversation_id"},
        {"name": "hinglish_labelled_synthetic", "category": "hinglish_labelled", "paths": [str(hi_syn)],
         "language": "hi-en", "source_type": "synthetic", "turns_field": "turns",
         "conversation_id_field": "conversation_id"},
    ]
    return sources, downstream


def make_tiny_model(dst: Path, base: str) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    cfg = AutoConfig.from_pretrained(base)
    cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.intermediate_size = 32, 2, 2, 64
    torch.manual_seed(0)
    model = AutoModelForMaskedLM.from_config(cfg)
    model.save_pretrained(dst)
    tok.save_pretrained(dst)


def write_config(work: Path, sources: list[dict], downstream: list[dict], model_ref: str) -> Path:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {"processed_dir": str(work / "processed"), "prepared_dir": str(work / "prepared"),
                    "tokenized_cache_dir": str(work / "tokenized"), "reports_dir": str(work / "reports"),
                    "output_root": str(work / "outputs")}
    cfg["model"] = {"name_or_path": model_ref, "tokenizer_name_or_path": None, "revision": None}
    cfg["sources"] = sources
    cfg["preprocessing"]["dedup"]["near_duplicate"]["enabled"] = True
    cfg["split"].update({"validation_ratio": 0.2, "test_ratio": 0.2})
    cfg["tokenization"] = {"max_seq_length": 64, "num_proc": 1}
    cfg["training"].update({
        "final_model_dir": str(work / "outputs" / "xlm-roberta-dapt" / "{experiment}"),
        "per_device_train_batch_size": 2, "per_device_eval_batch_size": 4, "gradient_accumulation_steps": 2,
        "eval_steps": 4, "logging_steps": 2, "dataloader_num_workers": 0, "early_stopping_patience": 2,
        "num_train_epochs": 1, "max_steps": 12, "save_total_limit": 2})
    cfg["smoke_test"] = {"max_train_chunks": 32, "max_eval_chunks": 16, "max_steps": 6, "eval_steps": 3,
                         "max_seq_length": 64}
    cfg["evaluation"].update({"batch_size": 8, "pll": {"enabled": True, "max_sequences_per_category": 2,
                                                       "max_seq_length": 32, "batch_size": 16}})
    final_mixed = str(work / "outputs" / "xlm-roberta-dapt" / "mixed")
    cfg["comparison"] = {"models": {"tiny_base": model_ref, "tiny_dapt": final_mixed}}
    cfg["downstream"].update({
        "sources": downstream, "split_dir": str(work / "downstream"), "output_dir": str(work / "outputs" / "downstream"),
        "models": {"tiny_base": model_ref, "tiny_dapt": final_mixed}, "seeds": [1], "num_train_epochs": 1,
        "max_seq_length": 64, "window_overlap": 16, "per_device_train_batch_size": 4, "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 1, "bootstrap_samples": 50, "min_test_conversations_per_class": 1,
        "class_balance": "oversample"})
    path = work / "self_check_config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


class Checker:
    def __init__(self, work: Path):
        self.work, self.results, self.logs = work, [], work / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, args: list[str], expect_fail: bool = False, expect_text: str | None = None) -> bool:
        log = self.logs / f"{len(self.results):02d}_{name.replace(' ', '_')}.log"
        t0 = time.time()
        proc = subprocess.run([sys.executable, *args], cwd=HERE, capture_output=True, text=True)
        log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
        ok = (proc.returncode != 0) if expect_fail else (proc.returncode == 0)
        if ok and expect_text:
            ok = expect_text in (proc.stdout + proc.stderr)
        self.record(name, ok, f"{time.time() - t0:.0f}s, log {log.name}")
        if not ok:
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-25:]
            print("      " + "\n      ".join(tail))
        return ok

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        self.record(name, bool(cond), detail)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def data_assertions(c: Checker, work: Path) -> None:
    prepared = {s: read_jsonl(work / "prepared" / f"{s}.jsonl") for s in ("train", "validation", "test")}
    texts = [d["text"] for s in prepared.values() for d in s]
    by_id = {d["conversation_id"]: d for s in prepared.values() for d in s}
    groups = {s: {d["group_id"] for d in docs} for s, docs in prepared.items()}
    c.check("no conversation shared between splits",
            not (groups["train"] & groups["validation"]) and not (groups["train"] & groups["test"])
            and not (groups["validation"] & groups["test"]))
    c.check("every category has validation data",
            {d["category"] for d in prepared["validation"]} == {"korean_scam", "korean_normal", "hinglish"})
    c.check("synthetic conversations only in train",
            all(d["source_type"] != "synthetic" for s in ("validation", "test") for d in prepared[s]))
    c.check("labels absent from prepared data",
            all("label" not in (d.get("metadata") or {}) and "label" not in d for s in prepared.values() for d in s))
    joined = "\n".join(texts)
    c.check("phone / Aadhaar / email / UPI / PAN / RRN redacted",
            "98765 43210" not in joined and "2345 6789 0123" not in joined and "ramesh.k@example.com" not in joined
            and "ramesh@okaxis" not in joined and "ABCDE1234F" not in joined and "900101-1234567" not in joined
            and "010-1234-5678" not in joined and "[PHONE]" in joined and "[AADHAAR_NUMBER]" in joined)
    c.check("money amounts preserved", "Rs 50,000" in joined)
    c.check("timestamps and noise tags removed", "(잡음)" not in joined and "[00:00:" not in joined)
    c.check("role names neutralised (no 'scammer'/'victim')",
            "scammer" not in joined and "victim" not in joined and "S1:" in joined)
    turns_doc = by_id.get("ko_scam_turns_0")
    c.check("turn records assembled in turn_index order", turns_doc is not None and
            [ln.split()[2] for ln in turns_doc["text"].splitlines()] == ["0", "1", "2", "3", "4"])
    removed = read_jsonl(work / "reports" / "removed_records.jsonl")
    reasons = {r["reason"] for r in removed}
    c.check("defects reported, not silently dropped",
            {"json_parse_error", "empty_raw_text", "missing_text_field", "exact_duplicate"} <= reasons,
            f"reasons={sorted(reasons)}")
    c.check("removed-records log has no text", all("text" not in r for r in removed))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_only", action="store_true")
    ap.add_argument("--real_model", action="store_true")
    ap.add_argument("--base_model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="dapt_self_check_"))
    print(f"Working in {work}")
    c = Checker(work)
    sources, downstream = make_fixtures(work)
    model_ref = args.base_model
    if not args.data_only and not args.real_model:
        print("Building a tiny random-weight XLM-R with the real tokenizer ...")
        make_tiny_model(work / "tiny-xlmr", args.base_model)
        model_ref = str(work / "tiny-xlmr")
    cfg = str(write_config(work, sources, downstream, model_ref))
    base = ["--config", cfg]

    c.run("inspect_data", ["inspect_data.py", *base])
    c.run("preprocess", ["preprocess.py", *base])
    c.run("prepare_dataset", ["prepare_dataset.py", *base])
    data_assertions(c, work)
    c.run("downstream-split", ["compare_models.py", *base, "downstream-split"])
    excl = [f"--set=exclusions.conversation_id_files=[{work / 'downstream' / 'test_conversation_ids.txt'}]",
            f"--set=exclusions.text_files=[{work / 'downstream' / 'test.jsonl'}]"]
    c.run("prepare_dataset with exclusions", ["prepare_dataset.py", *base, *excl])
    test_ids = set((work / "downstream" / "test_conversation_ids.txt").read_text().split())
    prepared_ids = {d["conversation_id"] for s in ("train", "validation", "test")
                    for d in read_jsonl(work / "prepared" / f"{s}.jsonl")}
    manifest = json.loads((work / "prepared" / "manifest.json").read_text())
    c.check("downstream test conversations excluded from DAPT", test_ids and not (test_ids & prepared_ids),
            f"{len(test_ids)} ids, excluded={manifest['exclusions']['excluded_documents']}")

    if not args.data_only:
        for exp in ("korean_only", "hinglish_only", "mixed"):
            c.run(f"train --dry_run {exp}", ["train_dapt.py", *base, "--experiment", exp, "--dry_run"])
        c.run("train --smoke_test", ["train_dapt.py", *base, "--experiment", "mixed", "--smoke_test"])
        smoke_final = work / "outputs" / "runs" / "mixed-smoke" / "final-model"
        c.check("smoke model saved with tokenizer", (smoke_final / "config.json").is_file()
                and (smoke_final / "dapt_info.json").is_file()
                and any(smoke_final.glob("tokenizer*")))

        # Interrupted run -> resume
        run_args = ["train_dapt.py", *base, "--experiment", "mixed", "--run_name", "resume-test",
                    "--set", "training.max_steps=24", "--set", "training.eval_steps=4",
                    "--set", "training.early_stopping_patience=0"]
        ckpt_dir = work / "outputs" / "runs" / "resume-test" / "checkpoints"
        log = (work / "logs" / "interrupted_run.log").open("w")
        proc = subprocess.Popen([sys.executable, *run_args], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + 900
        while time.time() < deadline and proc.poll() is None:
            if any((p / "trainer_state.json").is_file() for p in ckpt_dir.glob("checkpoint-*")):
                time.sleep(1.0)
                proc.kill()
                break
            time.sleep(0.5)
        proc.wait()
        log.close()
        interrupted = proc.returncode != 0
        c.check("training interrupted after first checkpoint", interrupted,
                "finished before it could be killed" if not interrupted else "killed")
        c.run("refuse to overwrite existing checkpoints", run_args, expect_fail=True, expect_text="already contains")
        c.run("refuse resume with changed settings", [*run_args, "--resume_from_checkpoint", "auto",
                                                      "--set", "training.learning_rate=1.0e-3"],
              expect_fail=True, expect_text="Cannot resume")
        c.run("resume from checkpoint", [*run_args, "--resume_from_checkpoint", "auto"])
        info_path = work / "outputs" / "xlm-roberta-dapt" / "mixed" / "dapt_info.json"
        steps = json.loads(info_path.read_text())["global_step"] if info_path.is_file() else None
        c.check("resumed run reached max_steps", steps == 24, f"global_step={steps}")

        c.run("evaluate_dapt (with PLL)", ["evaluate_dapt.py", *base, "--model",
                                           str(work / "outputs" / "xlm-roberta-dapt" / "mixed"), "--pll"])
        c.run("compare_models mlm", ["compare_models.py", *base, "mlm"])
        c.run("inference_mlm", ["inference_mlm.py", *base, "--model", str(work / "outputs" / "xlm-roberta-dapt" / "mixed"),
                                "--compare_with", model_ref, "--text", "Aap <mask> ho?", "--top_k", "3"])
        c.run("compare_models downstream", ["compare_models.py", *base, "downstream"])
        report = work / "reports" / "downstream" / "downstream_comparison.json"
        c.check("downstream report written", report.is_file())

    failed = [n for n, ok in c.results if not ok]
    print("\n" + ("ALL CHECKS PASSED" if not failed else f"{len(failed)} CHECK(S) FAILED: {failed}"))
    if failed or args.keep:
        print(f"Logs and outputs kept in {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

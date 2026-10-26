"""
Unit tests for the torch-free core logic. Run with either:
    python -m pytest tests/
    python tests/test_core.py
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dapt_utils as du  # noqa: E402
import preprocess  # noqa: E402


def test_config_override_and_scientific_floats(tmp_path=None):
    cfg = {"training": {"learning_rate": "5e-5"}, "sources": [{"paths": ["a"]}]}
    cfg = du._coerce_scientific(cfg)
    assert cfg["training"]["learning_rate"] == 5e-5
    du.apply_override(cfg, "training.learning_rate=3e-5")
    du.apply_override(cfg, "sources.0.paths=[x, y]")
    du.apply_override(cfg, "new.section.flag=true")
    assert cfg["training"]["learning_rate"] == 3e-5
    assert cfg["sources"][0]["paths"] == ["x", "y"]
    assert cfg["new"]["section"]["flag"] is True


def test_balanced_chunk_spans_cover_everything_without_tiny_tails():
    for n in (1, 7, 254, 255, 509, 1000, 12345):
        spans = du.balanced_chunk_spans(n, 254)
        assert spans[0][0] == 0 and spans[-1][1] == n
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
        sizes = [e - s for s, e in spans]
        assert max(sizes) <= 254 and max(sizes) - min(sizes) <= 1
    assert du.balanced_chunk_spans(0, 10) == []


def test_sliding_windows_overlap_and_coverage():
    spans = du.sliding_windows(1000, 254, 64)
    assert spans[0] == (0, 254) and spans[-1][1] == 1000
    assert all(b[0] - a[0] == 190 for a, b in zip(spans[:-1], spans[1:-1]))
    assert du.sliding_windows(10, 254, 64) == [(0, 10)]


def test_fixed_mask_is_deterministic_and_respects_special_tokens():
    ids = list(range(10, 210))
    stm = [1] + [0] * 198 + [1]
    a = du.fixed_mlm_mask(ids, stm, mask_token_id=4, vocab_size=1000, mlm_probability=0.15, seed_parts=("s", 1, "c#0"))
    b = du.fixed_mlm_mask(ids, stm, mask_token_id=4, vocab_size=1000, mlm_probability=0.15, seed_parts=("s", 1, "c#0"))
    c = du.fixed_mlm_mask(ids, stm, mask_token_id=4, vocab_size=1000, mlm_probability=0.15, seed_parts=("s", 1, "c#1"))
    assert a == b and a != c
    masked, labels = a
    assert labels[0] == -100 and labels[-1] == -100
    selected = [i for i, l in enumerate(labels) if l != -100]
    assert 10 <= len(selected) <= 55  # ~15% of 198
    assert all(labels[i] == ids[i] for i in selected)
    assert all(masked[i] == ids[i] for i in range(len(ids)) if labels[i] == -100)
    share_mask = sum(masked[i] == 4 for i in selected) / len(selected)
    assert share_mask > 0.5
    # a 1-token body still gets one masked position
    m, l = du.fixed_mlm_mask([0, 99, 2], [1, 0, 1], mask_token_id=4, vocab_size=1000, mlm_probability=0.01,
                             seed_parts=("x",))
    assert l == [-100, 99, -100]


def test_mixture_strategies():
    chunks = {"ko": [(i, 100) for i in range(90)], "hi": [(90 + i, 100) for i in range(10)]}
    t = du.mixture_targets({"ko": 9000, "hi": 1000}, "proportional")
    assert math.isclose(t["hi"], 0.1)
    t = du.mixture_targets({"ko": 9000, "hi": 1000}, "balanced")
    assert math.isclose(t["hi"], 0.5)
    t = du.mixture_targets({"ko": 9000, "hi": 1000}, "temperature", alpha=0.5)
    assert math.isclose(t["hi"], math.sqrt(1000) / (math.sqrt(1000) + math.sqrt(9000)))
    idx, rep = du.sample_mixture(chunks, "balanced", 0.5, None, None, 20, 1, 0)
    assert abs(rep["categories"]["hi"]["effective_share"] - 0.5) < 0.02
    assert rep["categories"]["hi"]["upsample_factor"] > 4.9
    assert any("repeated" in w for w in rep["warnings"])
    # cap: balanced wants 5x Hinglish, cap at 2x
    idx, rep = du.sample_mixture(chunks, "balanced", 0.5, None, None, 2, 1, 0)
    assert rep["categories"]["hi"]["capped_by_max_upsample_factor"]
    assert abs(rep["categories"]["hi"]["upsample_factor"] - 2.0) < 1e-9
    # proportional over 3 passes keeps the natural mix and uses every chunk
    idx, rep = du.sample_mixture(chunks, "proportional", 1.0, None, None, 10, 3, 0)
    assert len(idx) == 300 and Counter(idx)[0] == 3
    # determinism
    assert du.sample_mixture(chunks, "temperature", 0.5, None, None, 10, 2, 7)[0] == \
        du.sample_mixture(chunks, "temperature", 0.5, None, None, 10, 2, 7)[0]
    # custom weights must cover every category
    try:
        du.sample_mixture(chunks, "custom", 0.5, {"ko": 1.0}, None, 10, 1, 0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_balanced_eval_subset():
    chunks = {"ko": [(i, 50) for i in range(40)], "hi": [(40 + i, 50) for i in range(4)]}
    idx = du.balanced_subset(chunks, 0)
    cats = Counter("hi" if i >= 40 else "ko" for i in idx)
    assert cats == {"hi": 4, "ko": 4}


def test_group_split_is_deterministic_and_disjoint():
    groups = [f"g{i}" for i in range(50)]
    a, w = du.split_group_list(groups, 0.1, 0.1, (1, "x"))
    b, _ = du.split_group_list(list(reversed(groups)), 0.1, 0.1, (1, "x"))
    assert a == b and not w
    assert Counter(a.values()) == {"train": 40, "validation": 5, "test": 5}
    small, w = du.split_group_list(["a", "b"], 0.1, 0.1, (1, "y"))
    assert Counter(small.values())["train"] >= 1 and w


def test_assemble_turns_and_generated_ids():
    src = {"name": "s", "category": "c", "text_field": "text", "speaker_field": "spk", "turn_order_field": "t"}
    items = [{"file": "f", "index": i, "record": r} for i, r in enumerate([
        {"conversation_id": "c1", "t": 2, "spk": "B", "text": "two"},
        {"conversation_id": "c1", "t": 1, "spk": "A", "text": "one"},
        {"text": "standalone", "label": "scam"},
    ])]
    units = [du.record_to_unit(it, src, ["label"])[0] for it in items]
    docs, notes = du.assemble_documents(units)
    assert [t for _, t in docs[0]["turns"]] == ["one", "two"]
    assert docs[1]["conversation_id_generated"] and docs[1]["labels"] == {"label": "scam"}
    assert "label" not in docs[1]["metadata"]


def test_cleaner_preserves_content_and_redacts():
    pcfg = {"redaction": {"enabled": True, "types": {"phone_in": True, "aadhaar": True, "email": True}},
            "artifact_patterns": [{"name": "noise", "pattern": r"\[(?:noise|inaudible)\]", "flags": "i"}],
            "speaker_labels": {"mode": "neutralize"}}
    c = preprocess.TextCleaner(pcfg)
    text, n = c.format_turns([("Scammer", "jaldi jaldi paise bhejo"), ("Victim", "kitne?"), ("scammer", "50,000")])
    assert text == "S1: jaldi jaldi paise bhejo\nS2: kitne?\nS1: 50,000" and n == 2
    out, stats = c.clean("Call 98765 43210 [Noise] abhi, warna arrest!!  ")
    assert out == "Call [PHONE] abhi, warna arrest!!"
    assert stats["redacted:phone_in"] == 1 and stats["artifact:noise"] == 1
    assert c.clean("ㅋㅋㅋ 빨리빨리 이체하세요")[0] == "ㅋㅋㅋ 빨리빨리 이체하세요"


def test_near_duplicates():
    base = "aap ka account block ho jayega agar aap abhi payment nahi karte hain " * 5
    docs = [{"text": base}, {"text": base + " theek hai"}, {"text": "bilkul alag baat cheet hai yeh " * 6}]
    dups = preprocess.find_near_duplicates(docs, threshold=0.8)
    assert set(dups) == {1} and dups[1][0] == 0


def test_metrics_and_bootstrap():
    y = [0, 0, 1, 1, 1, 0]
    p = [0.1, 0.6, 0.7, 0.9, 0.4, 0.2]
    m = du.binary_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert math.isclose(m["precision"], 2 / 3) and math.isclose(m["recall"], 2 / 3)
    t = du.best_f1_threshold(y, p)
    assert du.binary_metrics(y, p, t)["f1"] >= m["f1"]
    bs = du.paired_bootstrap(y, p, [0.0, 0.0, 1.0, 1.0, 1.0, 0.0], "pr_auc", n_samples=200)
    assert bs["mean_delta"] > 0 and bs["valid_resamples"] > 100


def test_labels():
    pos, neg = du.LABEL_POSITIVE_DEFAULT, du.LABEL_NEGATIVE_DEFAULT
    assert du.normalise_binary_label("Scam", pos, neg) == 1
    assert du.normalise_binary_label(0, pos, neg) == 0
    assert du.normalise_binary_label("maybe", pos, neg) is None


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)

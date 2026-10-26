"""
mlm_utils.py - torch / transformers helpers shared by train_dapt.py, evaluate_dapt.py,
compare_models.py and inference_mlm.py.

* model / tokenizer loading (hub id or local folder)
* tokenisation + chunking of prepared documents (after the split, never before), cached on disk
* deterministic ("fixed") masking of evaluation chunks
* a collator that masks training batches dynamically and pads pre-masked evaluation batches
* token-weighted masked-LM loss / perplexity / accuracy and optional pseudo-log-likelihood
* version-tolerant construction of TrainingArguments and Trainer
* CUDA out-of-memory detection
"""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import hashlib
import inspect
import json
import logging
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import dapt_utils as du  # noqa: E402

LOGGER = logging.getLogger("dapt.mlm")
MIN_TRANSFORMERS = "4.46.0"
CHUNKER_VERSION = "balanced-chunks-v1"
CHUNK_COLUMNS = ["chunk_id", "doc_id", "group_id", "category", "language", "source_type", "n_tokens",
                 "input_ids", "attention_mask", "special_tokens_mask"]


# --------------------------------------------------------------------------- #
# Versions, devices, precision
# --------------------------------------------------------------------------- #
def check_library_versions() -> dict:
    import datasets
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) < Version(MIN_TRANSFORMERS):
        raise RuntimeError(f"transformers>={MIN_TRANSFORMERS} is required (found {transformers.__version__}). "
                           "Run: pip install -r requirements.txt")
    info = {"torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    LOGGER.info("torch %s | transformers %s | datasets %s | CUDA %s | GPU %s", info["torch"], info["transformers"],
                info["datasets"], info["cuda"], info["gpu"])
    return info


def eval_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(setting: str) -> dict:
    """Map config.training.mixed_precision to Trainer flags and an autocast dtype for evaluation."""
    if not torch.cuda.is_available():
        if setting in ("fp16", "bf16"):
            LOGGER.warning("mixed_precision=%s requested but CUDA is not available; using fp32.", setting)
        return {"fp16": False, "bf16": False, "autocast_dtype": None, "label": "fp32 (no CUDA)"}
    major, _minor = torch.cuda.get_device_capability(0)
    bf16_ok = major >= 8 and torch.cuda.is_bf16_supported()
    if setting == "no":
        return {"fp16": False, "bf16": False, "autocast_dtype": None, "label": "fp32"}
    if setting == "bf16":
        if not bf16_ok:
            raise ValueError("This GPU does not support bf16 (a T4 is compute capability 7.5). "
                             "Use training.mixed_precision=fp16 or auto.")
        return {"fp16": False, "bf16": True, "autocast_dtype": torch.bfloat16, "label": "bf16"}
    if setting == "fp16" or not bf16_ok:
        return {"fp16": True, "bf16": False, "autocast_dtype": torch.float16, "label": "fp16"}
    return {"fp16": False, "bf16": True, "autocast_dtype": torch.bfloat16, "label": "bf16"}


def autocast_ctx(device: torch.device, dtype):
    if dtype is not None and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def is_cuda_oom(exc: BaseException) -> bool:
    oom_types = tuple(t for t in (getattr(torch.cuda, "OutOfMemoryError", None), getattr(torch, "OutOfMemoryError", None))
                      if isinstance(t, type))
    if oom_types and isinstance(exc, oom_types):
        return True
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and "out of memory" in msg and ("cuda" in msg or "hip" in msg)


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Model and tokenizer loading
# --------------------------------------------------------------------------- #
def _hub_kwargs(cfg: dict, ref: str, resolved: str) -> dict:
    """Pass the configured revision only for the configured hub model, never for local folders."""
    if Path(resolved).exists():
        return {}
    configured = {du.cfg_get(cfg, "model.name_or_path"), du.cfg_get(cfg, "model.tokenizer_name_or_path")}
    rev = du.cfg_get(cfg, "model.revision")
    return {"revision": rev} if rev and ref in configured else {}


def load_tokenizer(cfg: dict, model_ref: str | None = None):
    from transformers import AutoTokenizer

    ref = model_ref or du.cfg_get(cfg, "model.tokenizer_name_or_path") or du.cfg_get(cfg, "model.name_or_path")
    resolved = du.resolve_model_ref(cfg, ref)
    tok = AutoTokenizer.from_pretrained(resolved, **_hub_kwargs(cfg, ref, resolved))
    if tok.mask_token_id is None or tok.pad_token_id is None:
        raise ValueError(f"Tokenizer {ref} has no mask/pad token; it is not an MLM tokenizer.")
    return tok


def load_mlm_model(cfg: dict, model_ref: str):
    from transformers import AutoModelForMaskedLM

    resolved = du.resolve_model_ref(cfg, model_ref)
    return AutoModelForMaskedLM.from_pretrained(resolved, **_hub_kwargs(cfg, model_ref, resolved))


def load_dapt_encoder(path: str):
    """Encoder only (no MLM head, no pooler) - the starting point for the XLM-R + GRU detector."""
    from transformers import AutoModel

    return AutoModel.from_pretrained(path, add_pooling_layer=False)


def special_template(tokenizer) -> tuple[list[int], list[int]]:
    """Special-token prefix/suffix the tokenizer adds to a single sequence (<s> ... </s> for XLM-R)."""
    probe = "abc xyz 123"
    with_sp = list(tokenizer(probe, add_special_tokens=True)["input_ids"])
    without = list(tokenizer(probe, add_special_tokens=False)["input_ids"])
    n = len(without)
    for start in range(len(with_sp) - n + 1):
        if with_sp[start:start + n] == without:
            return with_sp[:start], with_sp[start + n:]
    raise RuntimeError("Could not determine the tokenizer's special-token template")


def tokenizer_fingerprint(tokenizer) -> str:
    h = hashlib.sha1()
    for token, idx in sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1]):
        h.update(f"{idx}\t{token}\n".encode("utf-8"))
    prefix, suffix = special_template(tokenizer)
    h.update(json.dumps([prefix, suffix, tokenizer.mask_token_id, tokenizer.pad_token_id]).encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Tokenisation and chunking (after splitting)
# --------------------------------------------------------------------------- #
def load_prepared_manifest(cfg: dict) -> tuple[dict, Path]:
    prepared_dir = du.resolve_path(cfg, du.cfg_get(cfg, "paths.prepared_dir", "data/prepared"))
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found. Run prepare_dataset.py first.")
    return json.loads(manifest_path.read_text(encoding="utf-8")), prepared_dir


def chunk_documents(docs: list[dict], tokenizer, max_seq_length: int, num_proc: int | None = None,
                    desc: str = "Tokenising"):
    """One row per chunk. Each conversation is cut into equal-sized windows; chunks never mix
    conversations, so per-category and per-conversation bookkeeping stays exact."""
    from datasets import Dataset

    prefix, suffix = special_template(tokenizer)
    body = max_seq_length - len(prefix) - len(suffix)
    if body < 8:
        raise ValueError(f"max_seq_length={max_seq_length} leaves no room for text")
    if not docs:
        return None
    base = Dataset.from_list([{k: d[k] for k in ("doc_id", "group_id", "category", "language", "source_type", "text")}
                              for d in docs])

    def _chunk(batch: dict) -> dict:
        encoded = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
        out: dict[str, list] = {k: [] for k in CHUNK_COLUMNS}
        for i, ids in enumerate(encoded):
            for j, (s, e) in enumerate(du.balanced_chunk_spans(len(ids), body)):
                piece = list(ids[s:e])
                full = prefix + piece + suffix
                out["chunk_id"].append(f"{batch['doc_id'][i]}#{j}")
                for key in ("doc_id", "group_id", "category", "language", "source_type"):
                    out[key].append(batch[key][i])
                out["n_tokens"].append(len(piece))
                out["input_ids"].append(full)
                out["attention_mask"].append([1] * len(full))
                out["special_tokens_mask"].append([1] * len(prefix) + [0] * len(piece) + [1] * len(suffix))
        return out

    return base.map(_chunk, batched=True, batch_size=128, remove_columns=base.column_names,
                    num_proc=num_proc if num_proc and num_proc > 1 else None, desc=desc)


def tokenized_splits(cfg: dict, tokenizer, max_seq_length: int,
                     splits: tuple[str, ...] = ("train", "validation", "test")) -> tuple[dict, dict]:
    """Load chunked splits from the on-disk cache or build them. The cache key covers the prepared
    data fingerprint, the tokenizer vocabulary and the sequence length."""
    from datasets import load_from_disk

    manifest, prepared_dir = load_prepared_manifest(cfg)
    tok_fp = tokenizer_fingerprint(tokenizer)
    key = hashlib.sha1(f"{manifest['fingerprint']}|{tok_fp}|{max_seq_length}|{CHUNKER_VERSION}".encode()).hexdigest()[:16]
    cache_root = du.resolve_path(cfg, du.cfg_get(cfg, "paths.tokenized_cache_dir", "data/tokenized")) / key
    num_proc = int(du.cfg_get(cfg, "tokenization.num_proc", 1) or 1)
    out: dict = {}
    for split in splits:
        target = cache_root / split
        marker = cache_root / f"{split}.complete"
        if marker.is_file():
            out[split] = None if marker.read_text().strip() == "empty" else load_from_disk(str(target))
            continue
        docs = list(du.read_jsonl(prepared_dir / f"{split}.jsonl"))
        ds = chunk_documents(docs, tokenizer, max_seq_length, num_proc, desc=f"Tokenising {split}")
        if ds is None or len(ds) == 0:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("empty")
            out[split] = None
            continue
        missing = len({d["doc_id"] for d in docs}) - len(set(ds["doc_id"]))
        if missing:
            LOGGER.warning("%s: %d document(s) produced no tokens and were skipped", split, missing)
        if target.exists():
            shutil.rmtree(target)  # stale partial cache from an interrupted run
        ds.save_to_disk(str(target))
        marker.write_text("ok")
        out[split] = load_from_disk(str(target))
        LOGGER.info("%s: %d documents -> %d chunks (cached in %s)", split, len(docs), len(ds), target)
    info = {"cache_dir": str(cache_root), "tokenizer_fingerprint": tok_fp, "data_fingerprint": manifest["fingerprint"],
            "max_seq_length": max_seq_length}
    du.write_json(cache_root / "cache_info.json", info)
    return out, info


def chunks_by_category(ds, categories: list[str] | None = None) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, (cat, n) in enumerate(zip(list(ds["category"]), list(ds["n_tokens"]))):
        if categories is None or cat in categories:
            out[cat].append((i, int(n)))
    return dict(out)


def select_categories(ds, categories: list[str]):
    keep = [i for i, c in enumerate(list(ds["category"])) if c in set(categories)]
    return ds.select(keep)


def fixed_masked_dataset(ds, tokenizer, mlm_probability: float, mask_seed: int,
                         keep_columns: tuple[str, ...] = ("chunk_id", "category")):
    """Mask each chunk once, deterministically (seeded by chunk id), and store the labels."""
    mask_id, vocab_size = tokenizer.mask_token_id, len(tokenizer)

    def _mask(batch: dict) -> dict:
        ids_out, labels_out = [], []
        for cid, ids, stm in zip(batch["chunk_id"], batch["input_ids"], batch["special_tokens_mask"]):
            masked, labels = du.fixed_mlm_mask(ids, stm, mask_token_id=mask_id, vocab_size=vocab_size,
                                               mlm_probability=mlm_probability, seed_parts=("mlm-eval", mask_seed, cid))
            ids_out.append(masked)
            labels_out.append(labels)
        return {"input_ids": ids_out, "labels": labels_out}

    masked = ds.map(_mask, batched=True, batch_size=256, desc="Applying fixed evaluation masks")
    keep = ["input_ids", "attention_mask", "labels"] + [c for c in keep_columns if c in masked.column_names]
    return masked.select_columns(keep)


def eval_bundle(cfg: dict, tokenizer, split: str = "test", categories: list[str] | None = None,
                max_seq_length: int | None = None):
    """Chunked + fixed-masked evaluation data for one split (test falls back to validation if empty)."""
    msl = int(max_seq_length or du.cfg_get(cfg, "evaluation.max_seq_length")
              or du.cfg_get(cfg, "tokenization.max_seq_length", 256))
    data, info = tokenized_splits(cfg, tokenizer, msl, splits=(split,))
    ds, used = data[split], split
    if (ds is None or len(ds) == 0) and split == "test":
        LOGGER.warning("The test split is empty (split.test_ratio=0?). Falling back to validation; note that "
                       "checkpoints were selected on validation, so this comparison favours the DAPT model.")
        data, info = tokenized_splits(cfg, tokenizer, msl, splits=("validation",))
        ds, used = data["validation"], "validation"
    if ds is None or len(ds) == 0:
        raise RuntimeError(f"No evaluation data in split {split!r}")
    if categories:
        ds = select_categories(ds, categories)
        if len(ds) == 0:
            raise RuntimeError(f"No {split} chunks for categories {categories}")
    masked = fixed_masked_dataset(ds, tokenizer, float(du.cfg_get(cfg, "evaluation.mlm_probability", 0.15)),
                                  int(du.cfg_get(cfg, "evaluation.mask_seed", 1234)))
    return ds, masked, {**info, "split": used}


# --------------------------------------------------------------------------- #
# Collation
# --------------------------------------------------------------------------- #
def pad_premasked(input_ids: list, attention_mask: list | None, labels: list, pad_token_id: int,
                  multiple: int | None = 8) -> dict:
    length = max(len(x) for x in input_ids)
    if multiple:
        length = int(math.ceil(length / multiple) * multiple)
    bsz = len(input_ids)
    ids = torch.full((bsz, length), pad_token_id, dtype=torch.long)
    att = torch.zeros((bsz, length), dtype=torch.long)
    lab = torch.full((bsz, length), -100, dtype=torch.long)
    for i in range(bsz):
        n = len(input_ids[i])
        ids[i, :n] = torch.as_tensor(input_ids[i], dtype=torch.long)
        att[i, :n] = torch.as_tensor(attention_mask[i] if attention_mask is not None else [1] * n, dtype=torch.long)
        lab[i, :n] = torch.as_tensor(labels[i], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


class MLMCollator:
    """Training batches (no `labels`): dynamic masking by DataCollatorForLanguageModeling, so each
    epoch sees different masked positions. Evaluation batches (with `labels`): already masked
    deterministically, only padded - validation loss is then comparable across evaluations."""

    def __init__(self, tokenizer, mlm_probability: float, pad_to_multiple_of: int | None = 8):
        from transformers import DataCollatorForLanguageModeling

        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.dynamic = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability,
                                                       pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")

    def __call__(self, features: list[dict]) -> dict:
        if features and "labels" in features[0]:
            return pad_premasked([f["input_ids"] for f in features], [f.get("attention_mask") for f in features]
                                 if "attention_mask" in features[0] else None,
                                 [f["labels"] for f in features], self.pad_token_id, self.pad_to_multiple_of)
        keep = ("input_ids", "attention_mask", "special_tokens_mask")
        return self.dynamic([{k: f[k] for k in keep if k in f} for f in features])


# --------------------------------------------------------------------------- #
# Evaluation maths
# --------------------------------------------------------------------------- #
@torch.no_grad()
def masked_lm_scores(model, masked_ds, pad_token_id: int, batch_size: int = 16, autocast_dtype=None,
                     group_column: str = "category") -> dict:
    """Token-weighted masked-LM cross-entropy, exp(loss) and top-1 accuracy, per group and overall.

    Only the masked positions go through the LM head (vocabulary 250k), which keeps memory low.
    """
    device = eval_device()
    model.to(device)
    model.eval()
    lm_head = getattr(model, "lm_head", None)
    base = model.base_model if lm_head is not None else None
    groups = list(masked_ds[group_column]) if group_column in masked_ds.column_names else ["all"] * len(masked_ds)
    lengths = [len(x) for x in masked_ds["input_ids"]]
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    acc: dict[str, dict] = defaultdict(lambda: {"loss_sum": 0.0, "tokens": 0, "correct": 0, "chunks": 0})
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        rows = masked_ds[idx]
        batch = pad_premasked(rows["input_ids"], rows["attention_mask"], rows["labels"], pad_token_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        sel = labels != -100
        with autocast_ctx(device, autocast_dtype):
            if base is not None:
                hidden = base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])[0]
                logits = lm_head(hidden[sel])
            else:
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[sel]
        logits = logits.float()
        target = labels[sel]
        loss = F.cross_entropy(logits, target, reduction="none")
        correct = (logits.argmax(dim=-1) == target).float()
        row_of = sel.nonzero(as_tuple=True)[0]
        n_rows = len(idx)
        loss_rows = torch.zeros(n_rows, device=device).index_add_(0, row_of, loss)
        corr_rows = torch.zeros(n_rows, device=device).index_add_(0, row_of, correct)
        tok_rows = torch.bincount(row_of, minlength=n_rows)
        for j, i in enumerate(idx):
            for key in (groups[i], "ALL"):
                a = acc[key]
                a["loss_sum"] += float(loss_rows[j])
                a["correct"] += float(corr_rows[j])
                a["tokens"] += int(tok_rows[j])
                a["chunks"] += 1
    out = {}
    for key, a in acc.items():
        mean = a["loss_sum"] / a["tokens"] if a["tokens"] else float("nan")
        out[key] = {"chunks": a["chunks"], "masked_tokens": a["tokens"], "mlm_loss": mean,
                    "mlm_perplexity": du.safe_exp(mean) if a["tokens"] else float("nan"),
                    "top1_accuracy": a["correct"] / a["tokens"] if a["tokens"] else float("nan")}
    return out


@torch.no_grad()
def pseudo_log_likelihood(model, chunk_ds, tokenizer, per_category: int, max_seq_length: int,
                          batch_size: int = 64, seed: int = 0, autocast_dtype=None) -> dict:
    """Pseudo-log-likelihood: mask every token once, sum log p(token | rest). PPPL = exp(-PLL/N).
    A well-defined score for MLMs (Salazar et al., 2020); cost = one forward pass per token."""
    device = eval_device()
    model.to(device)
    model.eval()
    lm_head = getattr(model, "lm_head", None)
    base = model.base_model if lm_head is not None else None
    mask_id = tokenizer.mask_token_id
    cats = list(chunk_ds["category"])
    cids = list(chunk_ds["chunk_id"])
    chosen: dict[str, list[int]] = defaultdict(list)
    for cat in sorted(set(cats)):
        idx = sorted((i for i, c in enumerate(cats) if c == cat), key=lambda i: cids[i])
        du.rng_for(seed, "pll", cat).shuffle(idx)
        chosen[cat] = idx[:per_category]
    acc: dict[str, dict] = defaultdict(lambda: {"pll": 0.0, "tokens": 0, "sequences": 0})
    for cat, idx_list in chosen.items():
        for i in idx_list:
            row = chunk_ds[i]
            ids, stm = list(row["input_ids"]), list(row["special_tokens_mask"])
            n_pre = next((k for k, v in enumerate(stm) if v == 0), len(stm))
            n_suf = next((k for k, v in enumerate(reversed(stm)) if v == 0), 0)
            body = ids[n_pre:len(ids) - n_suf][: max(1, max_seq_length - n_pre - n_suf)]
            seq = torch.tensor(ids[:n_pre] + body + ids[len(ids) - n_suf:], device=device)
            positions = list(range(n_pre, n_pre + len(body)))
            for b in range(0, len(positions), batch_size):
                pos = torch.tensor(positions[b:b + batch_size], device=device)
                rows = torch.arange(len(pos), device=device)
                inp = seq.unsqueeze(0).repeat(len(pos), 1)
                inp[rows, pos] = mask_id
                with autocast_ctx(device, autocast_dtype):
                    if base is not None:
                        hidden = base(input_ids=inp, attention_mask=torch.ones_like(inp))[0]
                        logits = lm_head(hidden[rows, pos])
                    else:
                        logits = model(input_ids=inp, attention_mask=torch.ones_like(inp)).logits[rows, pos]
                logp = torch.log_softmax(logits.float(), dim=-1)[rows, seq[pos]]
                for key in (cat, "ALL"):
                    acc[key]["pll"] += float(logp.sum())
                    acc[key]["tokens"] += len(pos)
            for key in (cat, "ALL"):
                acc[key]["sequences"] += 1
    return {k: {"sequences": a["sequences"], "tokens": a["tokens"],
                "pll_per_token": a["pll"] / a["tokens"] if a["tokens"] else float("nan"),
                "pseudo_perplexity": du.safe_exp(-a["pll"] / a["tokens"]) if a["tokens"] else float("nan")}
            for k, a in acc.items()}


# --------------------------------------------------------------------------- #
# Trainer construction (tolerant to small API differences between versions)
# --------------------------------------------------------------------------- #
_REQUIRED_ARGS = ("output_dir", "eval_strategy", "save_strategy", "per_device_train_batch_size", "learning_rate")


def build_training_arguments(**kwargs: Any):
    from transformers import TrainingArguments

    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    if "eval_strategy" not in fields and "evaluation_strategy" in fields and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    unknown = sorted(k for k in kwargs if k not in fields)
    missing_core = [k for k in _REQUIRED_ARGS if k in unknown]
    if missing_core:
        raise RuntimeError(f"This transformers version lacks required TrainingArguments {missing_core}; "
                           "install the versions in requirements.txt")
    if unknown:
        LOGGER.warning("Ignoring TrainingArguments not supported by this transformers version: %s", unknown)
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in fields})


def make_trainer(tokenizer=None, trainer_cls=None, **kwargs: Any):
    from transformers import Trainer

    cls = trainer_cls or Trainer
    params = inspect.signature(cls.__init__).parameters
    if tokenizer is not None:
        if "processing_class" in params:
            kwargs["processing_class"] = tokenizer
        elif "tokenizer" in params:
            kwargs["tokenizer"] = tokenizer
    return cls(**kwargs)


def model_card(info: dict) -> str:
    cats = ", ".join(info.get("categories", []))
    return f"""---
library_name: transformers
base_model: {info.get('base_model')}
tags: [xlm-roberta, fill-mask, domain-adaptive-pretraining]
---
# XLM-R domain-adapted ({info.get('experiment')})

Continued masked-language-model pretraining of `{info.get('base_model')}` on conversation transcripts
(categories: {cats}). Tokenizer unchanged. Scam labels were not used.

* best validation MLM loss: {info.get('best_metric')}
* training steps: {info.get('global_step')}
* data fingerprint: {info.get('data_fingerprint')}

Load:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
tok = AutoTokenizer.from_pretrained("<this folder>")
mlm = AutoModelForMaskedLM.from_pretrained("<this folder>")
encoder = AutoModel.from_pretrained("<this folder>", add_pooling_layer=False)  # for downstream models
```

This is an intermediate checkpoint for fine-tuning; it is not a scam detector by itself.
"""

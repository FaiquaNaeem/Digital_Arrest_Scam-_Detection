"""
dapt_utils.py - shared, dependency-light helpers for the DAPT pipeline.

Deliberately free of torch / transformers imports so that inspect_data.py,
preprocess.py and prepare_dataset.py run on any machine that has PyYAML
(numpy and scikit-learn are imported lazily where needed).

Contents
--------
* configuration loading, CLI overrides and validation
* logging setup
* JSON / JSONL I/O and stable hashing
* raw source discovery and loading (JSONL, JSON, CSV/TSV, TXT)
* normalisation of raw records into the common schema
* group-level train/validation/test splitting
* multilingual mixture sampling (proportional / balanced / temperature / custom)
* chunk-boundary maths and deterministic ("fixed") MLM masking
* downstream classification metrics and paired bootstrap
"""
from __future__ import annotations

import csv
import datetime as _dt
import glob
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from importlib import metadata as _metadata
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

LOGGER = logging.getLogger("dapt")

SAMPLING_STRATEGIES = ("proportional", "balanced", "temperature", "custom")
PRECISION_CHOICES = ("auto", "fp16", "bf16", "no")
FORMAT_BY_EXT = {
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".json": "json",
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "txt",
}
ALL_FORMATS = ("jsonl", "json", "csv", "tsv", "txt")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
_SCI_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)[eE][+-]?\d+$")


def _coerce_scientific(node: Any) -> Any:
    """PyYAML (YAML 1.1) reads `5e-5` as a *string*. Convert such strings to float."""
    if isinstance(node, dict):
        return {k: _coerce_scientific(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_coerce_scientific(v) for v in node]
    if isinstance(node, str) and _SCI_NUMBER.match(node.strip()):
        return float(node)
    return node


def parse_override_value(raw: str) -> Any:
    value = yaml.safe_load(raw) if raw.strip() != "" else ""
    return _coerce_scientific(value)


def apply_override(cfg: dict, item: str) -> None:
    """Apply one `dotted.key=value` override. Integers index into lists."""
    if "=" not in item:
        raise ValueError(f"Override must look like section.key=value, got {item!r}")
    key, raw = item.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise ValueError(f"Empty override key in {item!r}")
    value = parse_override_value(raw)
    node: Any = cfg
    for part in parts[:-1]:
        if isinstance(node, list):
            node = node[int(part)]
        else:
            if part not in node or node[part] is None:
                node[part] = {}
            node = node[part]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def load_config(path: str | os.PathLike, overrides: Iterable[str] | None = None) -> dict:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_path} must contain a YAML mapping at the top level")
    cfg = _coerce_scientific(cfg)
    for item in overrides or []:
        apply_override(cfg, item)
    cfg["_meta"] = {"config_path": str(cfg_path), "base_dir": str(cfg_path.parent)}
    problems = validate_config(cfg)
    if problems:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(problems))
    return cfg


def cfg_get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return default if node is None else node


def validate_config(cfg: dict) -> list[str]:
    problems: list[str] = []
    sources = cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        problems.append("`sources` must be a non-empty list")
        sources = []
    names = set()
    categories = set()
    for i, src in enumerate(sources):
        if not isinstance(src, dict):
            problems.append(f"sources[{i}] must be a mapping")
            continue
        for key in ("name", "category", "paths"):
            if not src.get(key):
                problems.append(f"sources[{i}] is missing `{key}`")
        if src.get("name") in names:
            problems.append(f"duplicate source name {src.get('name')!r}")
        names.add(src.get("name"))
        categories.add(src.get("category"))
        if src.get("paths") is not None and not isinstance(src.get("paths"), list):
            problems.append(f"sources[{i}].paths must be a list")
        mode = src.get("txt_mode", "file")
        if mode not in ("file", "line", "blank_line"):
            problems.append(f"sources[{i}].txt_mode must be file|line|blank_line")
    experiments = cfg.get("experiments") or {}
    if not experiments:
        problems.append("`experiments` must define at least one experiment")
    for name, exp in experiments.items():
        cats = (exp or {}).get("categories") or []
        if not cats:
            problems.append(f"experiment {name!r} lists no categories")
        unknown = [c for c in cats if c not in categories]
        if unknown:
            problems.append(f"experiment {name!r} uses unknown categories {unknown}; known: {sorted(categories)}")
    active = cfg_get(cfg, "project.experiment")
    if active and active not in experiments:
        problems.append(f"project.experiment={active!r} is not defined under `experiments`")
    strategy = cfg_get(cfg, "sampling.strategy", "temperature")
    if strategy not in SAMPLING_STRATEGIES:
        problems.append(f"sampling.strategy must be one of {SAMPLING_STRATEGIES}")
    val = float(cfg_get(cfg, "split.validation_ratio", 0.1))
    test = float(cfg_get(cfg, "split.test_ratio", 0.1))
    if not (0 < val < 1) or not (0 <= test < 1) or val + test >= 1:
        problems.append("split ratios must satisfy 0 < validation_ratio, 0 <= test_ratio, validation+test < 1")
    msl = int(cfg_get(cfg, "tokenization.max_seq_length", 256))
    if not (16 <= msl <= 512):
        problems.append("tokenization.max_seq_length must be between 16 and 512 for XLM-R")
    prec = cfg_get(cfg, "training.mixed_precision", "auto")
    if prec not in PRECISION_CHOICES:
        problems.append(f"training.mixed_precision must be one of {PRECISION_CHOICES}")
    mlm_p = float(cfg_get(cfg, "training.mlm_probability", 0.15))
    if not (0 < mlm_p < 1):
        problems.append("training.mlm_probability must be in (0, 1)")
    return problems


def base_dir(cfg: dict) -> Path:
    return Path(cfg.get("_meta", {}).get("base_dir", "."))


def resolve_path(cfg: dict, value: str | os.PathLike) -> Path:
    """Relative paths in the config are resolved against the config file's folder."""
    p = Path(os.path.expandvars(str(value))).expanduser()
    return p if p.is_absolute() else (base_dir(cfg) / p)


def resolve_model_ref(cfg: dict, ref: str) -> str:
    """Return a local path if `ref` exists locally (relative to the config), else the hub id unchanged."""
    if ref is None:
        raise ValueError("model reference is None")
    candidate = resolve_path(cfg, ref)
    if candidate.exists():
        return str(candidate)
    return str(ref)


def format_template(value: str, **kwargs: Any) -> str:
    return str(value).format(**kwargs)


def config_snapshot(cfg: dict) -> dict:
    """Config without internal keys, safe to dump."""
    return json.loads(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, default=str))


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(cfg: dict | None, script_name: str, log_dir: str | os.PathLike | None = None,
                  level: str = "INFO") -> Path:
    if log_dir is None:
        reports = cfg_get(cfg or {}, "paths.reports_dir", "reports")
        log_dir = resolve_path(cfg, reports) / "logs" if cfg else Path("reports/logs")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"{script_name}_{stamp}.log"
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    LOGGER.info("Logging to %s", log_file)
    return log_file


# --------------------------------------------------------------------------- #
# I/O and hashing
# --------------------------------------------------------------------------- #
def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_int(*parts: Any) -> int:
    """Deterministic 60-bit integer from arbitrary parts (independent of PYTHONHASHSEED)."""
    joined = "\x1f".join(str(p) for p in parts)
    return int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:15], 16)


def rng_for(*parts: Any) -> random.Random:
    return random.Random(stable_int(*parts))


def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _atomic_write(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        write_fn(fh)
    os.replace(tmp, path)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    count = 0

    def _write(fh):
        nonlocal count
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1

    _atomic_write(Path(path), _write)
    return count


def write_json(path: str | os.PathLike, obj: Any) -> None:
    _atomic_write(Path(path), lambda fh: json.dump(obj, fh, ensure_ascii=False, indent=2, default=str))


def write_text(path: str | os.PathLike, text: str) -> None:
    _atomic_write(Path(path), lambda fh: fh.write(text))


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def package_versions(names: Iterable[str] = ("torch", "transformers", "datasets", "accelerate",
                                              "tokenizers", "numpy", "scikit-learn", "pyyaml")) -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in names:
        try:
            out[name] = _metadata.version(name)
        except _metadata.PackageNotFoundError:
            out[name] = None
    return out


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if math.isnan(v):
                return "n/a"
            if math.isinf(v):
                return "inf"
            return f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.1f}"
        if isinstance(v, int) and not isinstance(v, bool):
            return f"{v:,}"
        return str(v).replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def percentiles(values: list[float], qs: Iterable[float] = (0, 50, 90, 99, 100)) -> dict:
    if not values:
        return {f"p{int(q)}": None for q in qs}
    ordered = sorted(values)
    out = {}
    for q in qs:
        k = (len(ordered) - 1) * q / 100.0
        lo, hi = math.floor(k), math.ceil(k)
        val = ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)
        out[f"p{int(q)}"] = round(val, 2)
    return out


# --------------------------------------------------------------------------- #
# Source discovery and raw loading
# --------------------------------------------------------------------------- #
def discover_source_files(cfg: dict, source: dict) -> tuple[list[Path], list[str]]:
    """Expand a source's `paths` (files, directories or glob patterns) into concrete files."""
    forced = source.get("format")
    allowed = {f.lower().lstrip(".") for f in (source.get("formats") or ALL_FORMATS)}
    files: list[Path] = []
    missing: list[str] = []
    for pattern in source.get("paths") or []:
        target = resolve_path(cfg, pattern)
        if any(ch in str(pattern) for ch in "*?["):
            matches = [Path(m) for m in sorted(glob.glob(str(target), recursive=True))]
        elif target.is_dir():
            matches = sorted(p for p in target.rglob("*") if p.is_file())
        elif target.is_file():
            matches = [target]
        else:
            missing.append(str(pattern))
            continue
        kept = []
        for m in matches:
            if not m.is_file() or m.name.startswith("."):
                continue
            fmt = forced or FORMAT_BY_EXT.get(m.suffix.lower())
            if fmt and (forced or fmt in allowed):
                kept.append(m)
        if not kept:
            missing.append(f"{pattern} (no files with extensions for {sorted(allowed)})")
        files.extend(kept)
    seen, unique = set(), []
    for f in files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique, missing


def display_path(cfg: dict, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir(cfg).resolve()))
    except ValueError:
        return str(path)


def file_format(source: dict, path: Path) -> str:
    return source.get("format") or FORMAT_BY_EXT.get(path.suffix.lower(), "unknown")


def iter_file_records(cfg: dict, source: dict, path: Path) -> Iterator[dict]:
    """Yield raw items {"file", "index", "record"} or {"file", "index", "error"} for one file.

    File-level failures (undecodable file, invalid JSON document) are yielded as a single
    item with index -1 and an "error" key so callers can report instead of crashing midway.
    """
    rel = display_path(cfg, path)
    fmt = file_format(source, path)
    enc = source.get("encoding") or "utf-8-sig"
    text_field = source.get("text_field", "text")
    conv_field = source.get("conversation_id_field", "conversation_id")
    try:
        if fmt == "jsonl":
            with path.open(encoding=enc) as fh:
                for line_no, line in enumerate(fh):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        yield {"file": rel, "index": line_no, "error": f"json_parse_error: {exc.msg}"}
                        continue
                    if isinstance(obj, str):
                        obj = {text_field: obj}
                    if not isinstance(obj, dict):
                        yield {"file": rel, "index": line_no, "error": "record_not_an_object"}
                        continue
                    yield {"file": rel, "index": line_no, "record": obj}
        elif fmt == "json":
            with path.open(encoding=enc) as fh:
                data = json.load(fh)
            key = source.get("json_records_key")
            if key:
                if not isinstance(data, dict) or key not in data:
                    raise ValueError(f"json_records_key {key!r} not found in {rel}")
                data = data[key]
            if isinstance(data, dict):
                list_values = [v for v in data.values()
                               if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
                if text_field in data or source.get("turns_field") in data:
                    data = [data]
                elif len(list_values) == 1:
                    data = list_values[0]
                else:
                    raise ValueError(f"{rel}: cannot locate the record list; set `json_records_key`")
            if not isinstance(data, list):
                raise ValueError(f"{rel}: expected a list of records")
            for i, obj in enumerate(data):
                if isinstance(obj, str):
                    obj = {text_field: obj}
                if not isinstance(obj, dict):
                    yield {"file": rel, "index": i, "error": "record_not_an_object"}
                    continue
                yield {"file": rel, "index": i, "record": obj}
        elif fmt in ("csv", "tsv"):
            csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
            delimiter = source.get("csv_delimiter") or ("\t" if fmt == "tsv" else ",")
            with path.open(encoding=enc, newline="") as fh:
                reader = csv.DictReader(fh, delimiter=delimiter)
                if reader.fieldnames is None:
                    return
                for i, row in enumerate(reader):
                    row = {(k or "").strip(): v for k, v in row.items() if k is not None}
                    yield {"file": rel, "index": i, "record": row}
        elif fmt == "txt":
            content = path.read_text(encoding=enc)
            mode = source.get("txt_mode", "file")
            if mode == "file":
                conv_id = str(Path(rel).with_suffix(""))
                yield {"file": rel, "index": 0, "record": {text_field: content, conv_field: conv_id}}
            elif mode == "line":
                for i, line in enumerate(content.splitlines()):
                    if line.strip():
                        yield {"file": rel, "index": i, "record": {text_field: line}}
            else:  # blank_line
                blocks = re.split(r"\n\s*\n", content)
                for i, block in enumerate(blocks):
                    if block.strip():
                        yield {"file": rel, "index": i, "record": {text_field: block}}
        else:
            yield {"file": rel, "index": -1, "error": f"unsupported_format: {fmt}"}
    except UnicodeDecodeError as exc:
        hint = " (Korean files saved by Excel are often cp949; set `encoding: cp949` for this source)"
        yield {"file": rel, "index": -1, "error": f"decode_error[{enc}]: {exc.reason}{hint}"}
    except (json.JSONDecodeError, ValueError, KeyError, csv.Error) as exc:
        yield {"file": rel, "index": -1, "error": f"file_error: {exc}"}


def iter_source_records(cfg: dict, source: dict) -> Iterator[dict]:
    files, _missing = discover_source_files(cfg, source)
    for path in files:
        yield from iter_file_records(cfg, source, path)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def record_to_unit(item: dict, source: dict, label_fields: Iterable[str]) -> tuple[dict | None, str | None]:
    """Normalise one raw record into a "unit" (a list of (speaker, text) turns + metadata).

    Returns (unit, None) on success or (None, reason) when the record carries no usable text.
    """
    rec = item["record"]
    text_field = source.get("text_field", "text")
    turns_field = source.get("turns_field")
    speaker_field = source.get("speaker_field")
    conv_field = source.get("conversation_id_field", "conversation_id")
    lang_field = source.get("language_field", "language")
    stype_field = source.get("source_type_field", "source_type")
    order_field = source.get("turn_order_field")
    label_fields = list(label_fields)

    turns: list[tuple[str | None, str]] = []
    if turns_field and isinstance(rec.get(turns_field), list):
        t_text = source.get("turn_text_field", "text")
        t_spk = source.get("turn_speaker_field", "speaker")
        for turn in rec[turns_field]:
            if isinstance(turn, dict):
                txt, spk = turn.get(t_text), turn.get(t_spk)
            elif isinstance(turn, str):
                txt, spk = turn, None
            else:
                continue
            if txt is None:
                continue
            turns.append((None if spk in (None, "") else str(spk), str(txt)))
        if not turns:
            return None, "no_turn_text"
    else:
        if text_field not in rec:
            return None, "missing_text_field"
        raw = rec.get(text_field)
        if raw is None:
            return None, "null_text"
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            raw = str(raw)
        if not isinstance(raw, str):
            return None, "non_string_text"
        spk = rec.get(speaker_field) if speaker_field else None
        turns = [(None if spk in (None, "") else str(spk), raw)]

    if not any(t.strip() for _, t in turns):
        return None, "empty_raw_text"

    conv_raw = rec.get(conv_field)
    conv_id = None if conv_raw in (None, "") else str(conv_raw)
    skip = {text_field, conv_field, lang_field, stype_field, *label_fields}
    if turns_field:
        skip.add(turns_field)
    if speaker_field:
        skip.add(speaker_field)
    metadata = {k: _json_safe(v) for k, v in rec.items() if k not in skip}
    labels = {k: _json_safe(rec[k]) for k in label_fields if k in rec and rec[k] not in (None, "")}
    unit = {
        "source_name": source["name"],
        "category": source["category"],
        "file": item["file"],
        "index": item["index"],
        "conversation_id": conv_id,
        "turn_order": rec.get(order_field) if order_field else None,
        "language": str(rec.get(lang_field) or source.get("language") or "unknown"),
        "source_type": str(rec.get(stype_field) or source.get("source_type") or "unknown"),
        "turns": turns,
        "metadata": metadata,
        "labels": labels,
    }
    return unit, None


def _order_key(value: Any):
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(value))


def assemble_documents(units: list[dict], assemble: bool = True) -> tuple[list[dict], Counter]:
    """Group units that share a conversation_id (within a source) into conversation documents.

    Units without a conversation_id become their own document with a generated, stable id
    derived from file path and record index.
    """
    notes: Counter = Counter()
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for u in units:
        if u["conversation_id"] is None:
            cid = f"{u['file']}#{u['index']}"
            u = {**u, "_generated_id": True}
            key = cid
            notes["generated_conversation_id"] += 1
        else:
            cid = u["conversation_id"]
            key = cid if assemble else f"{cid}@@{u['file']}#{u['index']}"
        u = {**u, "_cid": cid}
        groups.setdefault(key, []).append(u)

    docs = []
    for key, members in groups.items():
        cid = members[0]["_cid"]
        if len(members) > 1:
            notes["multi_record_conversations"] += 1
            if all(m["turn_order"] is not None for m in members):
                members = sorted(members, key=lambda m: _order_key(m["turn_order"]))
            elif any(m["turn_order"] is not None for m in members):
                notes["partial_turn_order_kept_file_order"] += 1
        turns = [t for m in members for t in m["turns"]]
        langs = Counter(m["language"] for m in members)
        stypes = Counter(m["source_type"] for m in members)
        if len(langs) > 1:
            notes["mixed_language_within_conversation"] += 1
        if len(stypes) > 1:
            notes["mixed_source_type_within_conversation"] += 1
        labels: dict = {}
        label_conflicts = []
        for m in members:
            for k, v in m["labels"].items():
                if k in labels and str(labels[k]) != str(v):
                    label_conflicts.append(k)
                labels.setdefault(k, v)
        if label_conflicts:
            notes["label_conflict_within_conversation"] += 1
        meta = dict(members[0]["metadata"])
        doc_id = f"{members[0]['source_name']}::{key}"
        docs.append({
            "doc_id": doc_id,
            "conversation_id": cid,
            "conversation_id_generated": bool(members[0].get("_generated_id")),
            "category": members[0]["category"],
            "source_name": members[0]["source_name"],
            "language": langs.most_common(1)[0][0],
            "source_type": stypes.most_common(1)[0][0],
            "source_files": sorted({m["file"] for m in members}),
            "record_refs": [f"{m['file']}#{m['index']}" for m in members],
            "n_records": len(members),
            "turns": turns,
            "metadata": meta,
            "labels": labels,
            "label_conflicts": sorted(set(label_conflicts)),
        })
    return docs, notes


# --------------------------------------------------------------------------- #
# Text profiling
# --------------------------------------------------------------------------- #
def script_profile(text: str) -> Counter:
    """Counts of letters by script. Used to spot files filed under the wrong language."""
    counts: Counter = Counter()
    for ch in text:
        cp = ord(ch)
        if 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
            counts["hangul"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            counts["latin"] += 1
        elif ch.isdigit():
            counts["digit"] += 1
        elif ch.isalpha():
            counts["other_letter"] += 1
    return counts


SPEAKER_CANDIDATE_RE = re.compile(r"^\s*([^\n:：]{1,24}?)\s*[:：]\s+\S", re.MULTILINE)


def speaker_label_candidates(text: str) -> list[str]:
    return [m.group(1).strip() for m in SPEAKER_CANDIDATE_RE.finditer(text)]


# --------------------------------------------------------------------------- #
# Grouping and splitting
# --------------------------------------------------------------------------- #
def group_value(doc: dict, group_by: str) -> tuple[str, bool]:
    """Return (value, used_fallback) for the leakage-group field of a document."""
    if group_by in ("conversation_id", None, ""):
        return str(doc["conversation_id"]), False
    val = doc.get(group_by)
    if val in (None, ""):
        val = (doc.get("metadata") or {}).get(group_by)
    if val in (None, ""):
        return str(doc["conversation_id"]), True
    return str(val), False


def assign_group_ids(docs: list[dict], group_by: str = "conversation_id",
                     namespace: str = "source") -> Counter:
    """Set doc["group_id"]; every document of one group always lands in the same split."""
    notes: Counter = Counter()
    for doc in docs:
        value, fallback = group_value(doc, group_by)
        if fallback:
            notes[f"group_field_{group_by}_missing_used_conversation_id"] += 1
        prefix = doc["category"] if namespace == "category" else f"{doc['category']}::{doc['source_name']}"
        doc["group_id"] = f"{prefix}::{value}"
    return notes


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def split_group_list(group_ids: list[str], validation_ratio: float, test_ratio: float,
                     seed_parts: tuple, min_eval_groups: int = 1) -> tuple[dict[str, str], list[str]]:
    """Shuffle groups deterministically and cut them into test / validation / train.

    Small strata get at least `min_eval_groups` validation and test groups when that still
    leaves at least one training group; otherwise the shortfall is returned as a warning.
    """
    warnings: list[str] = []
    order = sorted(set(group_ids))
    rng_for(*seed_parts).shuffle(order)
    n = len(order)
    n_test = _round_half_up(test_ratio * n) if test_ratio > 0 else 0
    n_val = _round_half_up(validation_ratio * n) if validation_ratio > 0 else 0
    if test_ratio > 0 and n_test < min_eval_groups and n - n_val - min_eval_groups >= 1:
        n_test = min_eval_groups
    if validation_ratio > 0 and n_val < min_eval_groups and n - n_test - min_eval_groups >= 1:
        n_val = min_eval_groups
    while n_test + n_val > max(n - 1, 0) and (n_test or n_val):
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        else:
            n_val -= 1
    if validation_ratio > 0 and n_val == 0:
        warnings.append(f"only {n} group(s): no validation group could be held out")
    if test_ratio > 0 and n_test == 0:
        warnings.append(f"only {n} group(s): no test group could be held out")
    assignment = {}
    for i, gid in enumerate(order):
        assignment[gid] = "test" if i < n_test else ("validation" if i < n_test + n_val else "train")
    return assignment, warnings


# --------------------------------------------------------------------------- #
# Mixture sampling
# --------------------------------------------------------------------------- #
def mixture_targets(natural_tokens: dict[str, int], strategy: str, alpha: float = 0.5,
                    custom_weights: dict[str, float] | None = None) -> dict[str, float]:
    """Target share of training tokens per category.

    proportional : p_i = n_i / sum(n)            (alpha = 1)
    balanced     : p_i = 1 / k                   (alpha = 0)
    temperature  : p_i = n_i^alpha / sum(n^alpha) (XLM-R style smoothing, 0 < alpha < 1)
    custom       : p_i = w_i / sum(w)
    """
    cats = [c for c, n in natural_tokens.items() if n > 0]
    if not cats:
        raise ValueError("No category has any training tokens")
    if strategy == "proportional":
        raw = {c: float(natural_tokens[c]) for c in cats}
    elif strategy == "balanced":
        raw = {c: 1.0 for c in cats}
    elif strategy == "temperature":
        if alpha < 0:
            raise ValueError("sampling.alpha must be >= 0")
        raw = {c: float(natural_tokens[c]) ** alpha for c in cats}
    elif strategy == "custom":
        weights = custom_weights or {}
        missing = [c for c in cats if c not in weights]
        if missing:
            raise ValueError(f"sampling.custom_weights has no weight for {missing}")
        raw = {c: float(weights[c]) for c in cats}
        if any(v < 0 for v in raw.values()) or sum(raw.values()) <= 0:
            raise ValueError("sampling.custom_weights must be non-negative with a positive sum")
    else:
        raise ValueError(f"Unknown sampling strategy {strategy!r}")
    total = sum(raw.values())
    return {c: raw[c] / total for c in cats}


def _draw_to_budget(items: list[tuple[int, int]], target_tokens: float, rng: random.Random) -> tuple[list[int], int]:
    """Whole passes over `items` (shuffled) followed by a partial shuffled pass until the budget is met."""
    natural = sum(n for _, n in items)
    picked: list[int] = []
    got = 0
    if natural <= 0 or target_tokens <= 0:
        return picked, got
    for _ in range(int(target_tokens // natural)):
        perm = items[:]
        rng.shuffle(perm)
        picked.extend(i for i, _ in perm)
        got += natural
    if got < target_tokens:
        perm = items[:]
        rng.shuffle(perm)
        for idx, n in perm:
            if got >= target_tokens:
                break
            picked.append(idx)
            got += n
    return picked, got


def sample_mixture(chunks_by_category: dict[str, list[tuple[int, int]]], strategy: str, alpha: float,
                   custom_weights: dict | None, total_tokens: int | None, max_upsample_factor: float,
                   num_passes: int, seed: int) -> tuple[list[int], dict]:
    """Return dataset row indices for the training mixture plus a contribution report.

    `chunks_by_category[c]` is a list of (row_index, n_tokens). Each of `num_passes` passes is
    an independent draw, so downsampled categories show different chunks in different passes.
    """
    natural = {c: sum(n for _, n in items) for c, items in chunks_by_category.items()}
    empty = [c for c, n in natural.items() if n == 0]
    if empty:
        raise ValueError(f"Categories with no training tokens: {empty}")
    targets = mixture_targets(natural, strategy, alpha, custom_weights)
    budget = float(total_tokens) if total_tokens else float(sum(natural.values()))
    per_pass_target = {}
    capped = {}
    for c in natural:
        want = targets[c] * budget
        cap = max_upsample_factor * natural[c] if max_upsample_factor else float("inf")
        capped[c] = want > cap
        per_pass_target[c] = min(want, cap)

    all_indices: list[int] = []
    sampled_tokens: Counter = Counter()
    sampled_chunks: Counter = Counter()
    distinct: dict[str, set] = defaultdict(set)
    for p in range(max(1, int(num_passes))):
        for c in sorted(natural):
            idx, got = _draw_to_budget(chunks_by_category[c], per_pass_target[c], rng_for(seed, "mix", c, p))
            all_indices.extend(idx)
            sampled_tokens[c] += got
            sampled_chunks[c] += len(idx)
            distinct[c].update(idx)

    total_sampled = sum(sampled_tokens.values()) or 1
    passes = max(1, int(num_passes))
    natural_total = sum(natural.values())
    rows = {}
    warnings = []
    for c in sorted(natural):
        factor = sampled_tokens[c] / passes / natural[c]
        coverage = len(distinct[c]) / max(1, len(chunks_by_category[c]))
        rows[c] = {
            "natural_tokens": natural[c],
            "natural_chunks": len(chunks_by_category[c]),
            "natural_share": natural[c] / natural_total,
            "target_share": targets[c],
            "capped_by_max_upsample_factor": capped[c],
            "sampled_tokens_per_pass": sampled_tokens[c] / passes,
            "sampled_chunks_total": sampled_chunks[c],
            "effective_share": sampled_tokens[c] / total_sampled,
            "upsample_factor": factor,
            "distinct_chunk_coverage": coverage,
        }
        if factor >= 4:
            warnings.append(f"{c}: each chunk is repeated ~{factor:.1f}x per pass; the model may memorise "
                            f"these conversations. Consider a lower alpha or more {c} data.")
        if capped[c]:
            warnings.append(f"{c}: target share capped by max_upsample_factor={max_upsample_factor}; "
                            f"effective share {rows[c]['effective_share']:.1%} < target {targets[c]:.1%}.")
        if coverage < 0.5:
            warnings.append(f"{c}: only {coverage:.0%} of its chunks are used across all passes "
                            "(heavy downsampling).")
    report = {
        "strategy": strategy,
        "alpha": alpha if strategy == "temperature" else None,
        "custom_weights": custom_weights if strategy == "custom" else None,
        "token_budget_per_pass": budget,
        "num_passes": passes,
        "max_upsample_factor": max_upsample_factor,
        "total_training_chunks": len(all_indices),
        "total_training_tokens": total_sampled,
        "categories": rows,
        "warnings": warnings,
    }
    return all_indices, report


def balanced_subset(chunks_by_category: dict[str, list[tuple[int, int]]], seed: int) -> list[int]:
    """Downsample every category to the token count of the smallest one (for model selection)."""
    natural = {c: sum(n for _, n in items) for c, items in chunks_by_category.items() if items}
    if not natural:
        return []
    floor = min(natural.values())
    out: list[int] = []
    for c in sorted(natural):
        idx, _ = _draw_to_budget(chunks_by_category[c], floor, rng_for(seed, "eval-balance", c))
        out.extend(idx)
    return sorted(out)


# --------------------------------------------------------------------------- #
# Chunking and masking maths
# --------------------------------------------------------------------------- #
def balanced_chunk_spans(n_tokens: int, max_body: int) -> list[tuple[int, int]]:
    """Split n tokens into the fewest windows of <= max_body tokens, all of near-equal size.

    Equal-sized windows avoid tiny tail chunks without dropping any tokens.
    """
    if max_body <= 0:
        raise ValueError("max_body must be positive")
    if n_tokens <= 0:
        return []
    k = math.ceil(n_tokens / max_body)
    base, extra = divmod(n_tokens, k)
    spans, start = [], 0
    for j in range(k):
        size = base + (1 if j < extra else 0)
        spans.append((start, start + size))
        start += size
    return spans


def sliding_windows(n_tokens: int, max_body: int, overlap: int) -> list[tuple[int, int]]:
    """Overlapping windows for classification (every token appears in at least one window)."""
    if n_tokens <= 0:
        return []
    overlap = max(0, min(overlap, max_body - 1))
    step = max_body - overlap
    spans, start = [], 0
    while True:
        end = min(start + max_body, n_tokens)
        spans.append((start, end))
        if end >= n_tokens:
            break
        start += step
    return spans


def fixed_mlm_mask(input_ids: list[int], special_tokens_mask: list[int], *, mask_token_id: int,
                   vocab_size: int, mlm_probability: float, seed_parts: tuple,
                   mask_replace_prob: float = 0.8, random_replace_prob: float = 0.1) -> tuple[list[int], list[int]]:
    """Deterministic BERT-style corruption (80% <mask>, 10% random, 10% unchanged).

    Seeded per chunk, so every model and every evaluation sees exactly the same masked positions.
    At least one position is masked per chunk so no sequence contributes an empty loss.
    """
    import numpy as np

    rng = np.random.default_rng(stable_int(*seed_parts))
    ids = np.asarray(input_ids, dtype=np.int64)
    candidates = ~np.asarray(special_tokens_mask, dtype=bool)
    selected = (rng.random(len(ids)) < mlm_probability) & candidates
    if not selected.any() and candidates.any():
        selected[int(rng.choice(np.flatnonzero(candidates)))] = True
    labels = np.full(len(ids), -100, dtype=np.int64)
    labels[selected] = ids[selected]
    r = rng.random(len(ids))
    to_mask = selected & (r < mask_replace_prob)
    to_random = selected & (r >= mask_replace_prob) & (r < mask_replace_prob + random_replace_prob)
    masked = ids.copy()
    masked[to_mask] = mask_token_id
    n_random = int(to_random.sum())
    if n_random:
        masked[to_random] = rng.integers(0, vocab_size, size=n_random)
    return masked.tolist(), labels.tolist()


def safe_exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


# --------------------------------------------------------------------------- #
# Downstream classification metrics (scikit-learn imported lazily)
# --------------------------------------------------------------------------- #
LABEL_POSITIVE_DEFAULT = ["1", "scam", "fraud", "vishing", "digital_arrest", "true", "yes", "positive"]
LABEL_NEGATIVE_DEFAULT = ["0", "normal", "legit", "legitimate", "benign", "false", "no", "negative"]


def normalise_binary_label(value: Any, positives: Iterable[Any], negatives: Iterable[Any]) -> int | None:
    key = str(value).strip().casefold()
    if key in {str(v).strip().casefold() for v in positives}:
        return 1
    if key in {str(v).strip().casefold() for v in negatives}:
        return 0
    return None


def binary_metrics(y_true: list[int], probs: list[float], threshold: float) -> dict:
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)

    y_pred = [1 if p >= threshold else 0 for p in probs]
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1,
                                                       zero_division=0)
    both = len(set(y_true)) == 2
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    return {
        "threshold": threshold,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, probs)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, probs)) if both else float("nan"),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n": len(y_true),
        "n_positive": int(sum(y_true)),
    }


def best_f1_threshold(y_true: list[int], probs: list[float]) -> float:
    """Threshold that maximises F1 on validation data (ties -> highest threshold)."""
    if len(set(y_true)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in sorted(set(probs), reverse=True):
        f1 = binary_metrics(y_true, probs, t)["f1"]
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = t, f1
    return float(best_t)


def paired_bootstrap(y_true: list[int], probs_a: list[float], probs_b: list[float], metric: str,
                     threshold: float = 0.5, n_samples: int = 1000, seed: int = 0) -> dict:
    """Bootstrap CI for metric(B) - metric(A) on the same resampled conversations."""
    rng = random.Random(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        y = [y_true[i] for i in idx]
        if len(set(y)) < 2:
            continue
        a = binary_metrics(y, [probs_a[i] for i in idx], threshold)[metric]
        b = binary_metrics(y, [probs_b[i] for i in idx], threshold)[metric]
        deltas.append(b - a)
    if not deltas:
        return {"metric": metric, "valid_resamples": 0}
    deltas.sort()
    lo = deltas[int(0.025 * (len(deltas) - 1))]
    hi = deltas[int(0.975 * (len(deltas) - 1))]
    return {
        "metric": metric,
        "mean_delta": sum(deltas) / len(deltas),
        "ci95": [lo, hi],
        "p_delta_gt_0": sum(d > 0 for d in deltas) / len(deltas),
        "valid_resamples": len(deltas),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan")
    mu = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return mu, sd

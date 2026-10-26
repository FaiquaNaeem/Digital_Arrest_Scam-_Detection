"""
classifier_data.py - turns labelled Hinglish conversations into per-turn tensors for the
XLM-R + GRU multi-task model (train_classifier.py).

Reuses the fixed, leakage-safe conversation split already written by
`compare_models.py downstream-split` (data/downstream/{train,validation,test}.jsonl) so the
classifier's splits are identical to the ones the downstream encoder comparison used - same
conversations in train/validation/test, nothing re-shuffled.

Each turn is cleaned and PII-redacted with the exact same `preprocess.TextCleaner` settings
used for DAPT, and gets the same "S1:"/"S2:" neutralised speaker tag (first speaker in the
conversation -> S1, second distinct speaker -> S2), so the encoder sees text in the same
shape it was domain-adapted on.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

import dapt_utils as du

LOGGER = logging.getLogger("dapt.classifier_data")


def load_split_assignment(cfg: dict) -> dict[str, str]:
    """conversation_id -> 'train' | 'validation' | 'test', from the fixed downstream split."""
    ccfg = cfg["classifier"]
    split_dir = du.resolve_path(cfg, ccfg.get("split_dir", "data/downstream"))
    assignment: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path = split_dir / f"{split}.jsonl"
        if not path.is_file():
            raise SystemExit(f"{path} not found; run `python compare_models.py downstream-split` first.")
        for row in du.read_jsonl(path):
            assignment[row["conversation_id"]] = split
    return assignment


def _speaker_tag_map(turns: list[dict]) -> dict[str, str]:
    """Same rule as preprocess.TextCleaner.format_turns: first distinct speaker seen -> S1,
    second -> S2, in order of appearance."""
    mapping: dict[str, str] = {}
    for t in turns:
        key = str(t["speaker"]).strip().casefold()
        if key not in mapping:
            mapping[key] = f"S{len(mapping) + 1}"
    return mapping


def clean_and_tag_turns(cleaner, turns_raw: list[dict]) -> list[str]:
    """Shared with risk_score.py: same S1:/S2: tagging + redaction the classifier trained
    on, for scoring a new conversation (list of {"speaker", "text"})."""
    tag_map = _speaker_tag_map(turns_raw)
    cleaned = []
    for t in turns_raw:
        tag = tag_map[str(t["speaker"]).strip().casefold()]
        text, _ = cleaner.clean(f"{tag}: {t['text']}")
        cleaned.append(text)
    return cleaned


def load_conversations(cfg: dict) -> list[dict]:
    """Read the labelled synthetic jsonl and return one dict per conversation:
    {conversation_id, y (0/1), turns: [{text, tactics (list[str])}]}."""
    import preprocess

    ccfg = cfg["classifier"]
    dcfg = cfg.get("downstream") or {}
    path = du.resolve_path(cfg, ccfg["labelled_source"])
    if not path.is_file():
        raise SystemExit(f"{path} not found.")
    positives = {str(v).strip().casefold() for v in (dcfg.get("positive_labels") or du.LABEL_POSITIVE_DEFAULT)}
    cleaner = preprocess.TextCleaner(cfg.get("preprocessing") or {})
    tactic_labels = ccfg["tactic_labels"]
    tactic_index = {name: i for i, name in enumerate(tactic_labels)}

    out = []
    skipped_unknown_tactics = set()
    for row in du.read_jsonl(path):
        turns_raw = row.get("turns") or []
        if not turns_raw:
            continue
        cleaned_texts = clean_and_tag_turns(cleaner, turns_raw)
        turns = []
        for t, text in zip(turns_raw, cleaned_texts):
            labels = [0] * len(tactic_labels)
            for tac in t.get("tactics") or []:
                if tac in tactic_index:
                    labels[tactic_index[tac]] = 1
                else:
                    skipped_unknown_tactics.add(tac)
            turns.append({"text": text, "tactic_labels": labels})
        is_vishing = row.get("is_vishing")
        y = 1 if str(is_vishing).strip().casefold() in positives or is_vishing is True else 0
        out.append({"conversation_id": row["conversation_id"], "y": y, "turns": turns})
    if skipped_unknown_tactics:
        LOGGER.warning("Tactic labels found in data but not in classifier.tactic_labels, ignored: %s",
                       sorted(skipped_unknown_tactics))
    return out


class ConversationDataset(Dataset):
    """One example = one conversation. Turns are pre-tokenised at a fixed `max_seq_length`
    (padding="max_length") so the collate function only has to pad the turn dimension."""

    def __init__(self, conversations: list[dict], tokenizer, max_seq_length: int, max_turns: int,
                 n_tactics: int):
        self.n_tactics = n_tactics
        self.max_turns = max_turns
        self.examples = []
        over_cap = 0
        for conv in conversations:
            turns = conv["turns"][:max_turns]
            if len(conv["turns"]) > max_turns:
                over_cap += 1
            texts = [t["text"] for t in turns]
            enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_seq_length,
                            return_tensors="pt")
            tactic_targets = torch.tensor([t["tactic_labels"] for t in turns], dtype=torch.float32)
            self.examples.append({
                "conversation_id": conv["conversation_id"],
                "input_ids": enc["input_ids"],              # (n_turns, seq_len)
                "attention_mask": enc["attention_mask"],     # (n_turns, seq_len)
                "tactic_targets": tactic_targets,            # (n_turns, n_tactics)
                "n_turns": len(turns),
                "y": float(conv["y"]),
            })
        if over_cap:
            LOGGER.warning("%d conversation(s) exceeded max_turns=%d and were truncated.", over_cap, max_turns)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate(batch: list[dict]) -> dict:
    """Pad the turn dimension to the max number of turns in this batch (not the global cap)."""
    seq_len = batch[0]["input_ids"].shape[1]
    n_tactics = batch[0]["tactic_targets"].shape[1]
    c = max(ex["n_turns"] for ex in batch)
    b = len(batch)
    input_ids = torch.zeros(b, c, seq_len, dtype=torch.long)
    attention_mask = torch.zeros(b, c, seq_len, dtype=torch.long)
    tactic_targets = torch.zeros(b, c, n_tactics, dtype=torch.float32)
    turn_mask = torch.zeros(b, c, dtype=torch.float32)
    n_turns = torch.zeros(b, dtype=torch.long)
    y = torch.zeros(b, dtype=torch.float32)
    conversation_ids = []
    for i, ex in enumerate(batch):
        n = ex["n_turns"]
        input_ids[i, :n] = ex["input_ids"]
        attention_mask[i, :n] = ex["attention_mask"]
        tactic_targets[i, :n] = ex["tactic_targets"]
        turn_mask[i, :n] = 1.0
        n_turns[i] = n
        y[i] = ex["y"]
        conversation_ids.append(ex["conversation_id"])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "n_turns": n_turns,
            "tactic_targets": tactic_targets, "turn_mask": turn_mask, "y": y,
            "conversation_id": conversation_ids}


def build_datasets(cfg: dict, tokenizer) -> tuple[dict[str, ConversationDataset], dict]:
    ccfg = cfg["classifier"]
    assignment = load_split_assignment(cfg)
    conversations = load_conversations(cfg)
    by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    missing = 0
    for conv in conversations:
        split = assignment.get(conv["conversation_id"])
        if split is None:
            missing += 1
            continue
        by_split[split].append(conv)
    if missing:
        LOGGER.warning("%d conversation(s) in the labelled source have no split assignment "
                       "(not present in data/downstream/*.jsonl) and were dropped.", missing)
    n_tactics = len(ccfg["tactic_labels"])
    datasets = {s: ConversationDataset(rows, tokenizer, int(ccfg["max_seq_length"]), int(ccfg["max_turns"]),
                                       n_tactics)
               for s, rows in by_split.items()}
    info = {s: len(rows) for s, rows in by_split.items()}
    return datasets, info

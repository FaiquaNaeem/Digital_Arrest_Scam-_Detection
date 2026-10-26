"""
classifier_model.py - XLM-R + GRU multi-task scam detector.

Architecture, per dapt-xlmr's own README (section 12): a DAPT-adapted XLM-R encodes each
turn of a call independently (its <s> vector becomes the turn's representation), a GRU
reads those turn vectors in conversation order, and two heads sit on the GRU's output:
  - scam_head:   call-level scam probability (BCEWithLogits; sigmoid -> a percentage)
  - tactic_head: per-turn manipulation-tactic multi-label classification (BCEWithLogits;
                 a turn can show several tactics, or none - victim turns genuinely have none)

The GRU is unidirectional by default so the model is usable on a call in progress, not
just after it ends (no look-ahead into future turns).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


class XLMRGRUMultiTask(nn.Module):
    def __init__(self, encoder_path: str, n_tactics: int, gru_hidden: int = 256, dropout: float = 0.1,
                 bidirectional: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path, add_pooling_layer=False)
        hidden = self.encoder.config.hidden_size
        self.gru = nn.GRU(hidden, gru_hidden, batch_first=True, bidirectional=bidirectional)
        out = gru_hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.scam_head = nn.Linear(out, 1)
        self.tactic_head = nn.Linear(out, n_tactics)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        if enabled:
            self.encoder.gradient_checkpointing_enable()
        else:
            self.encoder.gradient_checkpointing_disable()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, n_turns: torch.Tensor) -> dict:
        b, c, l = input_ids.shape
        enc = self.encoder(input_ids=input_ids.view(b * c, l), attention_mask=attention_mask.view(b * c, l))
        turn_vec = enc.last_hidden_state[:, 0].view(b, c, -1)            # <s> vector of each turn
        packed = pack_padded_sequence(self.dropout(turn_vec), n_turns.cpu(), batch_first=True,
                                      enforce_sorted=False)
        seq, _ = self.gru(packed)
        seq, _ = pad_packed_sequence(seq, batch_first=True, total_length=c)   # (b, c, out)
        idx = torch.arange(b, device=seq.device)
        last = seq[idx, (n_turns - 1).to(seq.device)]                    # state after the last real turn
        return {"scam_logit": self.scam_head(self.dropout(last)).squeeze(-1),
                "tactic_logits": self.tactic_head(self.dropout(seq))}     # (b, c, n_tactics)


def compute_loss(outputs: dict, batch: dict, tactic_loss_weight: float = 1.0) -> dict:
    """BCEWithLogits on the scam head; BCEWithLogits on the tactic head, masked to real turns
    only (padded turn slots - not victim turns, which are real and legitimately tactic-free -
    never contribute to the loss)."""
    scam_loss = nn.functional.binary_cross_entropy_with_logits(outputs["scam_logit"], batch["y"])
    turn_mask = batch["turn_mask"]                                        # (b, c)
    per_turn_label_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["tactic_logits"], batch["tactic_targets"], reduction="none")   # (b, c, n_tactics)
    masked = per_turn_label_loss * turn_mask.unsqueeze(-1)
    denom = turn_mask.sum() * batch["tactic_targets"].shape[-1]
    tactic_loss = masked.sum() / denom.clamp(min=1.0)
    total = scam_loss + tactic_loss_weight * tactic_loss
    return {"loss": total, "scam_loss": scam_loss.detach(), "tactic_loss": tactic_loss.detach()}

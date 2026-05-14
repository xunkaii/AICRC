"""Step 4R-B v2 — BiGRU + Attention with auxiliary ambiguity-group head.

Drop-in API-compatible extension of `models.step4r_attention_rnn.Step4RBiGRUAttention`.

Architecture is identical to the baseline up to and including the attention
vector + posture concat, then forks into:
    - main head: 6-class logits over {C1..C6}                  (same as baseline)
    - aux head : 3-class logits over Step 2.5 ambiguity groups (NEW)
                   group 0 = {C2}
                   group 1 = {C1, C5, C6}   (within-group ambiguity)
                   group 2 = {C3, C4}       (pair, incl. C2-absorption case)

Aux target is *fully deterministic* from the true class label, so it adds no
new information — only coarse-to-fine representation organization (hard
auxiliary, not soft target).

Forward signature is kept compatible with the baseline (`logits, attn_w`)
so that downstream scripts (`calibrate_step4r_attention_temperature_v2.py`,
`generate_step4r_attention_schema_v2.py`) work unchanged. Training uses the
extended `forward_with_aux` method which also returns `logits_aux`.

Aux mapping helper:
    CLASS_TO_GROUP = [1, 0, 2, 2, 1, 1]    # indices match CLASSES = ["C1".."C6"]
    GROUPS         = ["confident_C2", "within_group_c1_c5_c6", "pair_c3_c4_extended"]
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.step4r_attention_rnn import LuongMultiplicativeAttention


# Aux group mapping: index = class index 0..5 (C1..C6), value = group 0..2
CLASS_TO_GROUP = [1, 0, 2, 2, 1, 1]
NUM_AUX_GROUPS = 3
AUX_GROUP_NAMES = ["confident_C2", "within_group_c1_c5_c6", "pair_c3_c4_extended"]


def class_idx_to_group_idx(y_idx: torch.Tensor) -> torch.Tensor:
    """Map (B,) class indices [0..5] -> (B,) aux group indices [0..2]."""
    mapping = torch.tensor(CLASS_TO_GROUP, device=y_idx.device, dtype=y_idx.dtype)
    return mapping[y_idx]


class Step4RBiGRUAttentionAux(nn.Module):
    """BiGRU + Luong attention + posture conditioning + main head + aux head.

    Backbone is structurally identical to Step4RBiGRUAttention (same params,
    same forward up to the classifier head). The aux head consumes the same
    pre-classifier feature vector `z` (attention_vec + posture one-hot) and
    produces 3-class group logits.
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 6,
        seq_len: int = 128,
        hidden_size: int = 64,
        num_layers: int = 2,
        rnn_dropout: float = 0.3,
        head_dropout: float = 0.3,
        posture_dim: int = 3,
        num_aux_groups: int = NUM_AUX_GROUPS,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_dropout = rnn_dropout
        self.head_dropout = head_dropout
        self.posture_dim = posture_dim
        self.num_aux_groups = num_aux_groups

        gru_dropout = rnn_dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=gru_dropout,
        )
        bi_hidden = hidden_size * 2  # 128

        self.attention = LuongMultiplicativeAttention(bi_hidden)

        # Main head: identical structure to baseline (same Linear / ReLU / Dropout / Linear).
        self.classifier = nn.Sequential(
            nn.Linear(bi_hidden + posture_dim, bi_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(bi_hidden, num_classes),
        )

        # Aux head: smaller, single hidden layer. Same input as main head.
        self.aux_head = nn.Sequential(
            nn.Linear(bi_hidden + posture_dim, bi_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(bi_hidden, num_aux_groups),
        )

    def _encode(self, x: torch.Tensor, posture: torch.Tensor):
        # x: (B, 6, 128) -> (B, 128, 6)
        x = x.transpose(1, 2)
        H, _ = self.gru(x)
        attn_vec, attn_w = self.attention(H)
        z = torch.cat([attn_vec, posture], dim=-1)
        return z, attn_w

    def forward_with_aux(
        self, x: torch.Tensor, posture: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits_main, attn_w, logits_aux). Used at training."""
        z, attn_w = self._encode(x, posture)
        logits_main = self.classifier(z)
        logits_aux = self.aux_head(z)
        return logits_main, attn_w, logits_aux

    def forward(
        self, x: torch.Tensor, posture: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """API-compatible with baseline. Returns (logits_main, attn_w)."""
        logits_main, attn_w, _ = self.forward_with_aux(x, posture)
        return logits_main, attn_w

    def get_config(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "rnn_dropout": self.rnn_dropout,
            "head_dropout": self.head_dropout,
            "posture_dim": self.posture_dim,
            "num_aux_groups": self.num_aux_groups,
        }

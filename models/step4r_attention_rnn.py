"""Step 4R-B — BiGRU + Luong-style multiplicative attention model (PyTorch).

Reimplements the structure of the Keras Attention.py reference layer in
PyTorch. The 7-step structure described in reports/step4_research_reframing.md
§5.2 is preserved:

    1. Linear projection of all hidden states
    2. last hidden state h_t serves as query
    3. multiplicative score = <h_t, W_a(H_t)>
    4. softmax over time -> attention_weights
    5. context = sum_t (attention_weights_t * H_t)
    6. attention_vector = tanh(W_c([context; h_t]))      (128-dim)
    7. forward returns (logits, attention_weights)

Forward signature:
    logits, attention_weights = model(x, posture_onehot)
where
    x: (B, 6, 128)        raw IMU after channel-wise train z-score
    posture_onehot: (B, 3) [SA, CA, HW] one-hot
returns
    logits: (B, num_classes=6)
    attention_weights: (B, T=128)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongMultiplicativeAttention(nn.Module):
    """Luong-style multiplicative attention with last-state query.

    Mirrors the structure of the source Keras Attention layer:
      - W_a projects each hidden state H_t -> H'_t   (step 1)
      - query = last timestep hidden state h_t       (step 2)
      - score_t = <h_t, H'_t>                        (step 3, multiplicative)
      - attention_weights = softmax(scores, dim=time) (step 4)
      - context = sum_t attention_weights_t * H_t    (step 5)
      - attention_vector = tanh(W_c([context; h_t])) (step 6)
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_c = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: (B, T, H)
        h_t = hidden_states[:, -1, :]                       # (B, H)
        H_proj = self.W_a(hidden_states)                    # (B, T, H)
        scores = torch.bmm(H_proj, h_t.unsqueeze(-1)).squeeze(-1)  # (B, T)
        attention_weights = F.softmax(scores, dim=-1)       # (B, T)
        context = torch.bmm(
            attention_weights.unsqueeze(1), hidden_states
        ).squeeze(1)                                        # (B, H)
        attention_vector = torch.tanh(
            self.W_c(torch.cat([context, h_t], dim=-1))
        )                                                   # (B, H)
        return attention_vector, attention_weights


class Step4RBiGRUAttention(nn.Module):
    """BiGRU encoder + Luong attention + posture conditioning + classifier.

    Posture conditioning: the (B, 3) posture one-hot is concatenated to the
    attention vector *after* the attention block, then fed to the head.
    Classifier head input dim = hidden_size * 2 + posture_dim = 128 + 3.
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

        # nn.GRU's `dropout` arg is only applied between stacked layers.
        gru_dropout = rnn_dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )
        bi_hidden = hidden_size * 2  # 128

        self.attention = LuongMultiplicativeAttention(bi_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(bi_hidden + posture_dim, bi_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(bi_hidden, num_classes),
        )

    def forward(
        self, x: torch.Tensor, posture: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 6, 128) -> (B, 128, 6)
        x = x.transpose(1, 2)
        H, _ = self.gru(x)                              # (B, 128, 128)
        attn_vec, attn_w = self.attention(H)            # (B, 128), (B, 128)
        z = torch.cat([attn_vec, posture], dim=-1)      # (B, 131)
        logits = self.classifier(z)                     # (B, num_classes)
        return logits, attn_w

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
        }

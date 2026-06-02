"""ResLSTM с multi-vector вниманием.

Авторитетный источник для класса `ResLSTM_Multi_Att`,
используемого в RAVDESS-экспериментах к статье DSPA 2026.

Архитектура — Variant A multi-vector attention (один скаляр-score per timestep
после Linear(num_att->1)), один LSTM с residual + второй LSTM, BN на residual
и после внимания.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ─────────────────────────────────────────────────────────────────────────────
# Версия модели. Менять при breaking change в архитектуре или forward-pass
# (что нарушит совместимость с ранее сохранёнными checkpoint'ами или с
# воспроизводимостью прежних studies). При v2 — не редактируем v1, а
# создаём lst_reslstm_v2.py, чтобы старые ноутбуки могли явно импортировать
# зафиксированную версию.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_VERSION = "v1"



class ResLSTM_Multi_Att(nn.Module):
    """ResLSTM с residual между LSTM1 и LSTM2 + Variant A multi-vector attention."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_att: int,
                 num_classes: int,
                 dropout_p: float = 0.1,
                 device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        # LSTM с одинаковыми размерностями для остаточного соединения
        self.lstm1 = nn.LSTM(input_size, input_size, num_layers,
                             bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers,
                             bidirectional=False, batch_first=True)

        # Multi-vector attention. Per-head scores -> Linear(num_att->1) -> softmax по T.
        # При num_att=1 это эквивалентно одному скалярному score-каналу.
        self.attention_vector = nn.Parameter(torch.empty(hidden_size, num_att))
        self.head_combine = nn.Linear(num_att, 1, bias=False)

        # BatchNorm на residual и после внимания
        self.bn_residual = nn.BatchNorm1d(input_size)
        self.bn = nn.BatchNorm1d(hidden_size)

        # Классификатор
        self.fc = nn.Linear(hidden_size, num_classes, device=self.device)
        self.drop = nn.Dropout(p=dropout_p)

        self.classes = num_classes
        self.num_att = num_att

        self.initialize_model_weights()

    def initialize_model_weights(self):
        for lstm, proj_size in [(self.lstm1, self.input_size),
                                (self.lstm2, self.hidden_size)]:
            for layer in range(self.num_layers):
                nn.init.xavier_normal_(getattr(lstm, f'weight_ih_l{layer}'))
                nn.init.orthogonal_(getattr(lstm, f'weight_hh_l{layer}'))
                bias_ih = getattr(lstm, f'bias_ih_l{layer}')
                bias_hh = getattr(lstm, f'bias_hh_l{layer}')
                nn.init.zeros_(bias_ih)
                nn.init.zeros_(bias_hh)
                # Forget gate bias = 1 (вторая четверть bias-вектора)
                with torch.no_grad():
                    bias_ih[proj_size: 2 * proj_size].fill_(1.0)
                    bias_hh[proj_size: 2 * proj_size].fill_(1.0)
        nn.init.xavier_normal_(self.attention_vector)
        nn.init.xavier_uniform_(self.head_combine.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def get_feature_vector(self, lstm_out):
        # (B, T, num_att)
        attention_scores_per_head = torch.matmul(lstm_out, self.attention_vector)
        # Linear projection num_att->1 (Variant A) → один score per timestep
        attention_scores = self.head_combine(attention_scores_per_head).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (B, T)
        feature_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return feature_vector, attention_weights

    def forward(self, x, lengths, return_lstm_out=False):
        batch_size = x.size(0)
        x_original = x.clone()

        # LSTM1 + residual
        h0 = torch.zeros(self.num_layers, batch_size, self.input_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.input_size).to(self.device)
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                        enforce_sorted=False)
        lstm1_out_packed, _ = self.lstm1(x_packed, (h0, c0))
        lstm1_out, _ = pad_packed_sequence(lstm1_out_packed, batch_first=True)

        if lstm1_out.size(1) != x.size(1):
            seq_len = min(lstm1_out.size(1), x.size(1))
            lstm1_out = lstm1_out[:, :seq_len, :]
            x_original = x_original[:, :seq_len, :]

        residual = lstm1_out + x_original
        # BatchNorm1d ожидает (B, F, T)
        residual_norm = self.bn_residual(residual.transpose(1, 2)).transpose(1, 2)

        # LSTM2
        h0_l2 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0_l2 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        residual_packed = pack_padded_sequence(residual_norm, lengths.cpu(),
                                               batch_first=True, enforce_sorted=False)
        lstm2_out_packed, _ = self.lstm2(residual_packed, (h0_l2, c0_l2))
        lstm2_out, _ = pad_packed_sequence(lstm2_out_packed, batch_first=True)

        feature_vector, attention_weights = self.get_feature_vector(lstm2_out)
        feature_vector = self.bn(feature_vector)
        logits = self.fc(self.drop(feature_vector))

        if return_lstm_out:
            return logits, attention_weights, lstm2_out
        return logits, attention_weights

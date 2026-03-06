
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
DEPRICATED
"""
class Encoder(nn.Module):
    def __init__(self, src_vocab: int, emb_dim: int, hid_dim: int, n_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor):
        emb = self.dropout(self.embedding(src_ids))  # [B, T, E]

        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)  # h: [L, B, H]
        return h


class Decoder(nn.Module):
    def __init__(self, tgt_vocab: int, emb_dim: int, hid_dim: int, n_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(tgt_vocab, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )
        self.out = nn.Linear(hid_dim, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, prev_token: torch.Tensor, state: torch.Tensor):
        emb = self.dropout(self.embedding(prev_token)).unsqueeze(1)  # [B, 1, E]
        out, new_state = self.gru(emb, state)                        # out: [B,1,H]
        logits = self.out(out.squeeze(1))                            # [B,V]
        return logits, new_state


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_id: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor, tgt_ids: torch.Tensor,
                teacher_forcing: float = 1.0):
        state = self.encoder(src_ids, src_lens)  # [L, B, H]

        B, Ttgt = tgt_ids.shape
        logits_seq = []

        prev = tgt_ids[:, 0]  # <bos>
        for t in range(1, Ttgt):
            logits, state = self.decoder.forward_step(prev, state)
            logits_seq.append(logits.unsqueeze(1))

            use_tf = (torch.rand(1).item() < teacher_forcing)
            prev = tgt_ids[:, t] if use_tf else logits.argmax(dim=-1)

        return torch.cat(logits_seq, dim=1)

    @torch.no_grad()
    def translate_greedy(self, src_ids: torch.Tensor, src_lens: torch.Tensor,
                         bos_id: int, eos_id: int, max_len: int = 60):
        state = self.encoder(src_ids, src_lens)  # [L,B,H]
        B = src_ids.size(0)

        prev = torch.full((B,), bos_id, device=src_ids.device, dtype=torch.long)
        out = [prev.unsqueeze(1)]

        for _ in range(max_len):
            logits, state = self.decoder.forward_step(prev, state)
            prev = logits.argmax(dim=-1)
            out.append(prev.unsqueeze(1))
            if (prev == eos_id).all():
                break

        return torch.cat(out, dim=1)

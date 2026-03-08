import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_padding_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    return (ids == pad_id).unsqueeze(1).unsqueeze(1)


def make_causal_mask(t: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0).unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        x = x.view(b, l, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, l, d_head = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, n_heads * d_head)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        q = self._split_heads(self.wq(q))
        k = self._split_heads(self.wk(k))
        v = self._split_heads(self.wv(v))

        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
        if past_v is not None:
            v = torch.cat([past_v, v], dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        ctx = torch.matmul(attn, v)
        ctx = self._merge_heads(ctx)

        out = self.wo(ctx)
        out = self.out_drop(out)
        if use_cache:
            return out, k, v
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.sa = MultiHeadAttention(d_model, n_heads, dropout=dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.sa(self.ln1(x), self.ln1(x), self.ln1(x), key_padding_mask=src_key_padding_mask)
        x = x + self.ff(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.sa = MultiHeadAttention(d_model, n_heads, dropout=dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ca = MultiHeadAttention(d_model, n_heads, dropout=dropout)

        self.ln3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        tgt_attn_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        layer_past: dict | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        past_self_k = None if layer_past is None else layer_past.get("sk")
        past_self_v = None if layer_past is None else layer_past.get("sv")

        y = self.ln1(x)
        if use_cache:
            sa_out, present_sk, present_sv = self.sa(
                y, y, y,
                attn_mask=tgt_attn_mask,
                key_padding_mask=tgt_key_padding_mask,
                past_k=past_self_k,
                past_v=past_self_v,
                use_cache=True,
            )
        else:
            sa_out = self.sa(
                self.ln1(x),
                self.ln1(x),
                self.ln1(x),
                attn_mask=tgt_attn_mask,
                key_padding_mask=tgt_key_padding_mask,
            )

        x = x + sa_out

        x = x + self.ca(
            self.ln2(x),
            mem,
            mem,
            key_padding_mask=src_key_padding_mask,
        )

        x = x + self.ff(self.ln3(x))
        
        if use_cache:
            present = {
                "sk": present_sk,
                "sv": present_sv,
            }
            return x, present
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.ln_f(x)


class Decoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_attn_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        past_key_values: list[dict] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        
        presents = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]

            if use_cache:
                x, present = layer(
                    x, memory,
                    tgt_attn_mask=tgt_attn_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    layer_past=layer_past,
                    use_cache=True,
                )
                presents.append(present)
            else:
                x = layer(
                    x,
                    memory,
                    tgt_attn_mask=tgt_attn_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    src_key_padding_mask=src_key_padding_mask,
                )
        return self.ln_f(x)


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, l = ids.shape
        idx = torch.arange(l, device=ids.device).unsqueeze(0).expand(b, l)
        return self.pos(idx)


@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    max_src_len: int
    max_tgt_len: int
    pad_id: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1


class TransformerSeq2Seq(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.src_tok_emb = nn.Embedding(cfg.src_vocab_size, cfg.d_model)
        self.tgt_tok_emb = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model)

        self.src_pos_emb = LearnedPositionalEmbedding(cfg.max_src_len, cfg.d_model)
        self.tgt_pos_emb = LearnedPositionalEmbedding(cfg.max_tgt_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.encoder = Encoder(cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
        self.decoder = Decoder(cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
        self.lm_head = nn.Linear(cfg.d_model, cfg.tgt_vocab_size, bias=False)

    def encode(self, src_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_key_padding_mask = make_padding_mask(src_ids, self.cfg.pad_id)

        x = self.src_tok_emb(src_ids) + self.src_pos_emb(src_ids)
        x = self.drop(x)

        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode(self, tgt_ids: torch.Tensor, memory: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        tgt_key_padding_mask = make_padding_mask(tgt_ids, self.cfg.pad_id)
        tgt_attn_mask = make_causal_mask(tgt_ids.size(1), device=tgt_ids.device)

        y = self.tgt_tok_emb(tgt_ids) + self.tgt_pos_emb(tgt_ids)
        y = self.drop(y)

        dec = self.decoder(
            y,
            memory,
            tgt_attn_mask=tgt_attn_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        logits = self.lm_head(dec)
        return logits

    def forward(self, src_ids: torch.Tensor, tgt_inp_ids: torch.Tensor) -> torch.Tensor:
        memory, src_key_padding_mask = self.encode(src_ids)
        logits = self.decode(tgt_inp_ids, memory, src_key_padding_mask)
        return logits

    @torch.no_grad()
    def beam_search_decode(
        self,
        src_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        unk_id: int,
        max_new_tokens: int,
        beam_size: int = 4,
        length_penalty: float = 0.6,
    ) -> torch.Tensor:
        self.eval()

        def normalized_score(candidate: dict) -> float:
            length = candidate["tokens"].size(1)
            if length_penalty == 0.0:
                return candidate["score"]
            return candidate["score"] / (length ** length_penalty)

        memory, src_key_padding_mask = self.encode(src_ids)

        beams = [
            {"tokens": torch.tensor([[bos_id]], dtype=torch.long, device=src_ids.device), "score": 0.0, "finished": False}
        ]

        for _ in range(max_new_tokens):
            if all(beam["finished"] for beam in beams):
                break

            all_candidates: list[dict] = []
            for beam in beams:
                if beam["finished"]:
                    all_candidates.append(beam)
                    continue

                logits = self.decode(beam["tokens"], memory, src_key_padding_mask)
                next_token_logits = logits[:, -1, :]
                # next_token_logits[:, unk_id] = -1e9
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

                for j in range(beam_size):
                    next_id = topk_ids[0, j].view(1, 1)
                    candidate = {
                        "tokens": torch.cat([beam["tokens"], next_id], dim=1),
                        "score": beam["score"] + topk_log_probs[0, j].item(),
                        "finished": next_id.item() == eos_id,
                    }
                    all_candidates.append(candidate)

            all_candidates.sort(key=normalized_score, reverse=True)
            beams = all_candidates[:beam_size]

        best_beam = max(beams, key=normalized_score)
        return best_beam["tokens"]


# Backward-compatible alias for an older typo in the class name.
TrasformerSeq2Seq = TransformerSeq2Seq

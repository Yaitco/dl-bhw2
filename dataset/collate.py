import torch
from torch import nn

def collate_fn(batch, pad_id, bos_id, eos_id):
    src_batch, tgt_batch = [], []

    for src, tgt in batch:
        src_batch.append(torch.tensor(src, dtype=torch.long))
        tgt = [bos_id] + tgt + [eos_id]
        tgt_batch.append(torch.tensor(tgt, dtype=torch.long))

    src_lens = torch.tensor([len(x) for x in src_batch], dtype=torch.long)

    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

    return src_padded, src_lens, tgt_padded
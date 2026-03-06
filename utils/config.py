from typing import Dict
import torch
import torch.nn as nn
from model.transformer import TransformerConfig, TransformerSeq2Seq

name_to_optimizer = {
    "Adam": torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'SGD': torch.optim.SGD,
}

def build_optimizer(config, model: nn.Module):
    config = dict(config) 
    name = config["name"]
    config.pop('name')

    optimizer = name_to_optimizer[name]

    return optimizer(model.parameters(), **config)


name_to_scheduler = {
    'StepLR': torch.optim.lr_scheduler.StepLR,
    'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
    'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
    'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
}

def build_scheduler(config: Dict, optimizer):
    config = dict(config) 
    name = config["name"]
    config.pop('name')

    scheduler = name_to_scheduler[name]

    return scheduler(optimizer, **config)

def build_model(config, src_vocab_size: int, tgt_vocab_size: int, pad_id: int):
    config = dict(config)
    name = str(config.pop("name", "TRANSFORMER")).upper()
    if name not in {"TRANSFORMER", "TRANSFORMER_SEQ2SEQ"}:
        raise ValueError(f"Unknown model name: {name}")

    transformer_cfg = TransformerConfig(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_src_len=int(config.get("max_src_len", 256)),
        max_tgt_len=int(config.get("max_tgt_len", 256)),
        pad_id=pad_id,
        d_model=int(config.get("d_model", 512)),
        n_heads=int(config.get("n_heads", 8)),
        n_layers=int(config.get("n_layers", 6)),
        d_ff=int(config.get("d_ff", 2048)),
        dropout=float(config.get("dropout", 0.1)),
    )
    return TransformerSeq2Seq(transformer_cfg)


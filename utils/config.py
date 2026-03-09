import math
from typing import Dict
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
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

def get_transformer_scheduler(optimizer, d_model, warmup_steps=4000):
    base_lr = float(optimizer.param_groups[0]["lr"])
    if base_lr <= 0:
        raise ValueError("Optimizer lr must be > 0 for transformer_warmup scheduler.")
    warmup_steps = int(warmup_steps)
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be > 0 for transformer_warmup scheduler.")

    def lr_lambda(step):
        step = max(step, 1)
        # Normalized Noam schedule:
        # - linear warmup up to 1.0 at `warmup_steps`
        # - inverse-sqrt decay afterwards
        # Effective lr is scaled by optimizer's current lr (base_lr).
        warmup_ratio = step / warmup_steps
        decay_ratio = (warmup_steps / step) ** 0.5
        return min(warmup_ratio, decay_ratio)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def build_scheduler(config: Dict, optimizer):
    config = dict(config) 
    name = config["name"]
    config.pop('name')

    normalized_name = str(name).lower().replace("-", "_")
    if normalized_name in {"transformer_warmup", "transformerwarmup", "noamlr"}:
        d_model = int(config.pop("d_model"))
        warmup_steps = int(config.pop("warmup_steps", 4000))
        return get_transformer_scheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

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


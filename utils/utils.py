import torch
import numpy as np
import random
from pathlib import Path
from comet_ml import Experiment
from sacrebleu.metrics import BLEU

def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state

def save_checkpoint(path, model, last_epoch, optimizer=None, meta=None, exp: Experiment | None = None):
    if meta is None:
        meta = {}
    meta['last_epoch'] = last_epoch
    meta['rng_state'] = get_rng_state()
    if exp is not None:
        meta['experiment_key'] = exp.get_key()
    ckpt = {"model": model.state_dict(), "meta": meta}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, path)

def trim_special_tokens(ids, bos_id, eos_id, pad_id):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()

    result = []

    for tok in ids:
        if tok == bos_id:
            continue
        if tok == eos_id:
            break
        if tok == pad_id:
            continue

        result.append(tok)

    return result

def seq2seq_loss(criterion, logits: torch.Tensor, tgt_out_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    B, T, V = logits.shape

    logits = logits.view(B * T, V)
    targets = tgt_out_ids.reshape(B * T)

    return criterion(logits, targets)

def corpus_bleu_sacrebleu(hypotheses, references):
    bleu = BLEU(tokenize="none")
    result = bleu.corpus_score(hypotheses, [references])
    return result

def average_checkpoints(checkpoint_paths, map_location="cpu"):
    checkpoint_paths = [str(p) for p in checkpoint_paths]
    if len(checkpoint_paths) == 0:
        raise ValueError("No checkpoints provided for averaging.")

    sum_state = {}
    float_dtypes = {}

    for idx, path in enumerate(checkpoint_paths):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        if idx == 0:
            for key, value in state.items():
                if torch.is_floating_point(value):
                    sum_state[key] = value.detach().clone().to(torch.float32)
                    float_dtypes[key] = value.dtype
                else:
                    sum_state[key] = value.detach().clone()
            continue

        for key, value in state.items():
            if torch.is_floating_point(value):
                sum_state[key] += value.detach().to(torch.float32)

    count = float(len(checkpoint_paths))
    for key in list(sum_state.keys()):
        if key in float_dtypes:
            sum_state[key] = (sum_state[key] / count).to(float_dtypes[key])

    return sum_state

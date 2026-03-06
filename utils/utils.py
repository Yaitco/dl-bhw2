import torch
import numpy as np
import random
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

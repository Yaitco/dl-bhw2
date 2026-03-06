import random
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from dataset.collate import collate_fn
from dataset.dataset import TestDataset, TranslationDataset, Vocabulary
from model.transformer import TransformerConfig, TransformerSeq2Seq
from train import train
from utils.config import build_optimizer, build_scheduler


def test_collate_fn(batch, source_vocab):
    src_batch = [torch.tensor(source_vocab.encode(line), dtype=torch.long) for line in batch]
    src_lens = torch.tensor([len(x) for x in src_batch], dtype=torch.long)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=source_vocab.pad_id)
    return src_padded, src_lens


def ids_to_text(ids, vocab: Vocabulary):
    result = []
    for token_id in ids:
        if token_id == vocab.eos_id:
            break
        if token_id in (vocab.pad_id, vocab.bos_id):
            continue
        if 0 <= token_id < len(vocab.idx_to_word):
            result.append(vocab.idx_to_word[token_id])
        else:
            result.append("<unk>")
    return " ".join(result)


@torch.no_grad()
def save_test_translations(
    model,
    source_vocab,
    target_vocab,
    test_path,
    output_path,
    device,
    test_loader_cfg,
    beam_size=4,
    max_new_tokens=80,
):
    model.eval()

    test_dataset = TestDataset(source_path=test_path)
    test_collate = partial(test_collate_fn, source_vocab=source_vocab)
    test_loader = DataLoader(test_dataset, collate_fn=test_collate, **test_loader_cfg)

    predictions = []
    for src_ids, _ in test_loader:
        src_ids = src_ids.to(device, non_blocking=True)

        for i in range(src_ids.size(0)):
            generated = model.beam_search_decode(
                src_ids=src_ids[i].unsqueeze(0),
                bos_id=target_vocab.bos_id,
                eos_id=target_vocab.eos_id,
                max_new_tokens=max_new_tokens,
                beam_size=beam_size,
            )
            predictions.append(ids_to_text(generated[0].tolist(), target_vocab))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))


def main(config=None):
    warnings.filterwarnings("ignore")

    if config is None:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    random_seed = config["random_seed"]
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    device = config["device"]
    beam_size = config["train"].get("beam_size", 4)
    max_new_tokens = config["train"].get("max_new_tokens", 80)
    max_grad_norm = config["train"].get("max_grad_norm", 1.0)

    source_vocab = Vocabulary(min_count=10)
    target_vocab = Vocabulary(min_count=10)

    with open(config["train_dataset"]["source_path"], "r", encoding="utf-8") as f:
        source = [line.strip("\n") for line in f]
        source_vocab.fit(source)

    with open(config["train_dataset"]["target_path"], "r", encoding="utf-8") as f:
        target = [line.strip("\n") for line in f]
        target_vocab.fit(target)

    train_dataset = TranslationDataset(
        source_path=config["train_dataset"]["source_path"],
        target_path=config["train_dataset"]["target_path"],
        source_vocab=source_vocab,
        target_vocab=target_vocab,
    )
    val_dataset = TranslationDataset(
        source_path=config["val_dataset"]["source_path"],
        target_path=config["val_dataset"]["target_path"],
        source_vocab=source_vocab,
        target_vocab=target_vocab,
    )

    collate = partial(
        collate_fn,
        pad_id=source_vocab.pad_id,
        bos_id=target_vocab.bos_id,
        eos_id=target_vocab.eos_id,
    )
    train_loader = DataLoader(train_dataset, collate_fn=collate, **config["train_loader"])
    val_loader = DataLoader(val_dataset, collate_fn=collate, **config["val_loader"])

    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "TRANSFORMER")).upper()
    if model_name not in {"TRANSFORMER", "TRANSFORMER_SEQ2SEQ"}:
        raise ValueError(f"Unknown model name: {model_name}")

    transformer_cfg = TransformerConfig(
        src_vocab_size=len(source_vocab),
        tgt_vocab_size=len(target_vocab),
        max_src_len=int(model_cfg.get("max_src_len", 256)),
        max_tgt_len=int(model_cfg.get("max_tgt_len", 256)),
        pad_id=source_vocab.pad_id,
        d_model=int(model_cfg.get("d_model", 512)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        n_layers=int(model_cfg.get("n_layers", 6)),
        d_ff=int(model_cfg.get("d_ff", 2048)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    model = TransformerSeq2Seq(transformer_cfg).to(device)

    optimizer = build_optimizer(config["optimizer"], model)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=target_vocab.pad_id)
    scheduler = build_scheduler(config["scheduler"], optimizer)

    experiment = None
    if config.get("comet"):
        from comet_ml import start

        experiment = start(project_name=config["comet_config"]["project_name"])
        experiment.log_asset("config.yaml")
        experiment.log_parameters(config)

    print(model)
    print(f"Number of params: {sum(p.numel() for p in model.parameters())}")
    best_bleu = train(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=val_loader,
        num_epochs=config["train"]["num_epochs"],
        device=device,
        pad_id=target_vocab.pad_id,
        bos_id=target_vocab.bos_id,
        eos_id=target_vocab.eos_id,
        tgt_vocab=target_vocab,
        experiment=experiment,
        beam_size=beam_size,
        max_new_tokens=max_new_tokens,
        max_grad_norm=max_grad_norm,
    )

    ckpt = torch.load(config["save_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    save_test_translations(
        model=model,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        test_path=config["test_dataset"]["source_path"],
        output_path=config["test_output"]["path"],
        device=device,
        test_loader_cfg=config["test_loader"],
        beam_size=config["inference"].get("beam_size", beam_size),
        max_new_tokens=config["inference"].get("max_len", max_new_tokens),
    )

    if experiment is not None:
        experiment.log_asset(config["test_output"]["path"])
        experiment.end()
    return best_bleu


if __name__ == "__main__":
    main()

import argparse
from functools import partial
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from dataset.dataset import TestDataset, Vocabulary
from model.transformer import TransformerConfig, TransformerSeq2Seq


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


def build_vocabs(config):
    source_vocab = Vocabulary()
    target_vocab = Vocabulary()

    with open(config["train_dataset"]["source_path"], "r", encoding="utf-8") as f:
        source = [line.strip("\n") for line in f]
    with open(config["train_dataset"]["target_path"], "r", encoding="utf-8") as f:
        target = [line.strip("\n") for line in f]

    source_vocab.fit(source)
    target_vocab.fit(target)
    return source_vocab, target_vocab


@torch.no_grad()
def run_inference(config, checkpoint_path=None, output_path=None):
    device = config["device"]
    source_vocab, target_vocab = build_vocabs(config)

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

    ckpt_path = checkpoint_path or config["inference"]["checkpoint_path"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_dataset = TestDataset(source_path=config["test_dataset"]["source_path"])
    test_loader = DataLoader(
        test_dataset,
        collate_fn=partial(test_collate_fn, source_vocab=source_vocab),
        **config["test_loader"],
    )

    predictions = []
    beam_size = int(config["inference"].get("beam_size", 4))
    max_new_tokens = int(config["inference"].get("max_len", config["test_output"].get("max_len", 80)))
    for src_ids, _ in test_loader:
        src_ids = src_ids.to(device, non_blocking=True)
        for i in range(src_ids.size(0)):
            pred = model.beam_search_decode(
                src_ids=src_ids[i].unsqueeze(0),
                bos_id=target_vocab.bos_id,
                eos_id=target_vocab.eos_id,
                max_new_tokens=max_new_tokens,
                beam_size=beam_size,
            )
            predictions.append(ids_to_text(pred[0].tolist(), target_vocab))

    out_path = output_path or config["inference"]["output_path"]
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))

    print(f"Inference finished. Saved {len(predictions)} translations to {out_file}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run standalone inference for test translations.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path override")
    parser.add_argument("--output", default=None, help="Output file path override")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_inference(cfg, checkpoint_path=args.checkpoint, output_path=args.output)

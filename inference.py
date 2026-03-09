import argparse
import json
from functools import partial
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.dataset import TestDataset, Vocabulary
from model.transformer import TransformerConfig, TransformerSeq2Seq
from utils.utils import average_checkpoints


def test_collate_fn(batch, source_vocab):
    src_batch = [torch.tensor(source_vocab.encode(line), dtype=torch.long) for line in batch]
    src_lens = torch.tensor([len(x) for x in src_batch], dtype=torch.long)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=source_vocab.pad_id)
    return src_padded, src_lens


def build_unk_replacements(copy_src_positions, src_token_ids, source_vocab: Vocabulary):
    if copy_src_positions is None:
        return None
    src_words = source_vocab.decode(src_token_ids.tolist())
    replacements = []
    for src_pos in copy_src_positions:
        if src_pos is None or src_pos < 0 or src_pos >= len(src_words):
            replacements.append(None)
            continue
        token = src_words[src_pos]
        if token in {"<pad>", "<bos>", "<eos>"}:
            replacements.append(None)
        else:
            replacements.append(token)
    return replacements


def ids_to_text(ids, vocab: Vocabulary, unk_replacements=None):
    result = []
    for idx, token_id in enumerate(ids):
        if token_id == vocab.eos_id:
            break
        if token_id in (vocab.pad_id, vocab.bos_id):
            continue
        if token_id == vocab.unk_id and unk_replacements is not None and idx < len(unk_replacements):
            replacement = unk_replacements[idx]
            if replacement is not None:
                result.append(replacement)
                continue
        if 0 <= token_id < len(vocab.idx_to_word):
            result.append(vocab.idx_to_word[token_id])
        else:
            result.append("<unk>")
    return " ".join(result)


def build_vocabs(config):
    vocab_min_count = int(config.get("vocab", {}).get("min_count", 10))
    source_vocab = Vocabulary(min_count=vocab_min_count)
    target_vocab = Vocabulary(min_count=vocab_min_count)

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
    max_new_tokens = int(config["inference"].get("max_len", config["test_output"].get("max_len", 80)))

    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "TRANSFORMER")).upper()
    if model_name not in {"TRANSFORMER", "TRANSFORMER_SEQ2SEQ"}:
        raise ValueError(f"Unknown model name: {model_name}")
    max_src_len = int(model_cfg.get("max_src_len", 256))
    max_tgt_len = int(model_cfg.get("max_tgt_len", 256))
    with open(config["test_dataset"]["source_path"], "r", encoding="utf-8") as f:
        test_src_max = max((len(line.strip().split()) for line in f), default=0)
    required_tgt_len = max_new_tokens + 1
    if max_src_len < test_src_max or max_tgt_len < required_tgt_len:
        raise ValueError(
            f"model positional limits are too small for inference: "
            f"max_src_len={max_src_len} (required >= {test_src_max}), "
            f"max_tgt_len={max_tgt_len} (required >= {required_tgt_len}). "
            f"Increase model.max_src_len/model.max_tgt_len in config.yaml."
        )

    transformer_cfg = TransformerConfig(
        src_vocab_size=len(source_vocab),
        tgt_vocab_size=len(target_vocab),
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        pad_id=source_vocab.pad_id,
        d_model=int(model_cfg.get("d_model", 512)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        n_layers=int(model_cfg.get("n_layers", 6)),
        d_ff=int(model_cfg.get("d_ff", 2048)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    model = TransformerSeq2Seq(transformer_cfg).to(device)

    avg_cfg = config.get("checkpoint_averaging", {})
    avg_enabled = bool(avg_cfg.get("enabled", False))
    avg_dir = Path(avg_cfg.get("dir", "checkpoint/avg"))
    avg_num_last = int(avg_cfg.get("num_last", 5))
    avg_manifest_path = avg_dir / "manifest.json"

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    elif avg_enabled:
        if avg_manifest_path.exists():
            payload = json.loads(avg_manifest_path.read_text(encoding="utf-8"))
            avg_paths = [avg_dir / rec["path"] for rec in payload.get("checkpoints", [])]
            avg_paths = [p for p in avg_paths if p.exists()][:avg_num_last]
        else:
            avg_paths = []
        if len(avg_paths) == 0:
            avg_paths = sorted(avg_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
            avg_paths = avg_paths[-avg_num_last:]
        if len(avg_paths) > 0:
            print(f"Using checkpoint averaging from {len(avg_paths)} checkpoints.")
            avg_state = average_checkpoints(avg_paths, map_location=device)
            model.load_state_dict(avg_state)
        else:
            ckpt_path = config["inference"]["checkpoint_path"]
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
    else:
        ckpt_path = config["inference"]["checkpoint_path"]
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
    replace_unk_with_attn_copy = bool(config["inference"].get("replace_unk_with_attn_copy", False))
    for src_ids, _ in tqdm(test_loader, desc="Inference"):
        src_ids = src_ids.to(device, non_blocking=True)
        for i in range(src_ids.size(0)):
            decode_out = model.beam_search_decode(
                src_ids=src_ids[i].unsqueeze(0),
                bos_id=target_vocab.bos_id,
                eos_id=target_vocab.eos_id,
                unk_id=target_vocab.unk_id,
                max_new_tokens=max_new_tokens,
                beam_size=beam_size,
                replace_unk_with_attn_src=replace_unk_with_attn_copy,
                return_copy_positions=replace_unk_with_attn_copy,
            )
            if replace_unk_with_attn_copy:
                pred, copy_src_positions = decode_out
                unk_replacements = build_unk_replacements(copy_src_positions, src_ids[i], source_vocab)
            else:
                pred = decode_out
                unk_replacements = None
            predictions.append(ids_to_text(pred[0].tolist(), target_vocab, unk_replacements=unk_replacements))

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

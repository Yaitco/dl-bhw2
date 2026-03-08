from sacrebleu.metrics import BLEU
from pathlib import Path
import json
import torch
import torch.nn as nn

from tqdm.auto import tqdm

from utils.utils import save_checkpoint, seq2seq_loss, trim_special_tokens


def corpus_bleu_sacrebleu(hypotheses, references):
    bleu = BLEU(tokenize="none")
    return float(bleu.corpus_score(hypotheses, [references]).score)


def training_epoch(
    model,
    optimizer,
    criterion,
    train_loader,
    tqdm_desc,
    device,
    pad_id,
    max_grad_norm: float = 1.0,
    scheduler=None,
    scheduler_name: str | None = None,
):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    scheduler_name = str(scheduler_name or "").lower().replace("-", "_")
    batch_step_schedulers = {"transformer_warmup", "transformerwarmup", "noamlr"}

    for src_ids, _, tgt_ids in tqdm(train_loader, desc=tqdm_desc):
        src_ids = src_ids.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)

        tgt_inp = tgt_ids[:, :-1]
        tgt_out = tgt_ids[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        logits = model(src_ids, tgt_inp)

        loss = seq2seq_loss(criterion, logits, tgt_out, pad_id)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None and scheduler_name in batch_step_schedulers:
            scheduler.step()

        tokens_in_batch = (tgt_out != pad_id).sum().item()
        total_loss += loss.item() * tokens_in_batch
        total_tokens += tokens_in_batch

    if scheduler is not None and scheduler_name not in {"plateau", *batch_step_schedulers}:
        scheduler.step()

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def validation_epoch(model, criterion, val_loader, tqdm_desc, device, pad_id):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src_ids, _, tgt_ids in tqdm(val_loader, desc=tqdm_desc):
        src_ids = src_ids.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)

        tgt_inp = tgt_ids[:, :-1]
        tgt_out = tgt_ids[:, 1:]

        logits = model(src_ids, tgt_inp)
        loss = seq2seq_loss(criterion, logits, tgt_out, pad_id)

        tokens_in_batch = (tgt_out != pad_id).sum().item()
        total_loss += loss.item() * tokens_in_batch
        total_tokens += tokens_in_batch

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def validation_epoch_bleu(
    model,
    val_loader,
    tqdm_desc,
    device,
    pad_id,
    bos_id,
    eos_id,
    beam_size,
    max_new_tokens,
    tgt_vocab,
    max_batches: int | None = None,
    max_samples: int | None = None,
):
    model.eval()

    hypotheses = []
    references = []
    processed_samples = 0

    for batch_idx, (src_ids, _, tgt_ids) in enumerate(tqdm(val_loader, desc=tqdm_desc)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        src_ids = src_ids.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)

        for i in range(src_ids.size(0)):
            if max_samples is not None and processed_samples >= max_samples:
                break
            pred = model.beam_search_decode(
                src_ids=src_ids[i].unsqueeze(0),
                bos_id=bos_id,
                eos_id=eos_id,
                unk_id=tgt_vocab.unk_id,
                max_new_tokens=max_new_tokens,
                beam_size=beam_size,
            )
            pred_tokens = trim_special_tokens(pred[0], bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)
            hyp = " ".join(tgt_vocab.decode(pred_tokens))
            hypotheses.append(hyp)

            ref_tokens = trim_special_tokens(tgt_ids[i], bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)
            ref = " ".join(tgt_vocab.decode(ref_tokens))
            references.append(ref)
            processed_samples += 1

        if max_samples is not None and processed_samples >= max_samples:
            break

    if len(hypotheses) == 0:
        return 0.0

    return corpus_bleu_sacrebleu(hypotheses, references)


def train(
    config,
    model: nn.Module,
    optimizer,
    scheduler,
    criterion,
    train_loader,
    test_loader,
    num_epochs,
    device,
    pad_id,
    bos_id,
    eos_id,
    tgt_vocab,
    experiment=None,
    beam_size: int = 4,
    max_new_tokens: int = 80,
    max_grad_norm: float = 1.0,
):
    best_bleu = None
    scheduler_name = str(config["scheduler"]["name"]).lower().replace("-", "_")
    label_smoothing = float(config.get("train", {}).get("label_smoothing", 0.0))
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError(f"train.label_smoothing must be in [0, 1). Got: {label_smoothing}")
    criterion_train = criterion
    if label_smoothing > 0.0:
        criterion_train = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)

    avg_cfg = config.get("checkpoint_averaging", {})
    avg_enabled = bool(avg_cfg.get("enabled", False))
    avg_dir = Path(avg_cfg.get("dir", "checkpoint/avg"))
    avg_num_last = int(avg_cfg.get("num_last", 5))
    avg_metric = str(avg_cfg.get("metric", "bleu")).lower()
    if avg_metric not in {"bleu", "loss"}:
        raise ValueError(f"checkpoint_averaging.metric must be 'bleu' or 'loss'. Got: {avg_metric}")
    avg_reset_on_start = bool(avg_cfg.get("reset_on_start", True))
    manifest_path = avg_dir / "manifest.json"
    avg_records = []

    bleu_max_batches = config.get("train", {}).get("bleu_max_batches")
    bleu_max_samples = config.get("train", {}).get("bleu_max_samples")
    bleu_max_batches = None if bleu_max_batches is None else int(bleu_max_batches)
    bleu_max_samples = None if bleu_max_samples is None else int(bleu_max_samples)

    if avg_enabled:
        avg_dir.mkdir(parents=True, exist_ok=True)
        if avg_reset_on_start:
            for old_path in avg_dir.glob("epoch_*.pt"):
                old_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion_train,
            train_loader=train_loader,
            tqdm_desc=f"Training {epoch}/{num_epochs}",
            device=device,
            pad_id=pad_id,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            scheduler_name=scheduler_name,
        )
        val_loss = validation_epoch(
            model=model,
            criterion=criterion,
            val_loader=test_loader,
            tqdm_desc=f"Validating {epoch}/{num_epochs} (loss)",
            device=device,
            pad_id=pad_id,
        )
        val_bleu = validation_epoch_bleu(
            model=model,
            val_loader=test_loader,
            tqdm_desc=f"Validating {epoch}/{num_epochs} (bleu)",
            device=device,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            beam_size=beam_size,
            max_new_tokens=max_new_tokens,
            tgt_vocab=tgt_vocab,
            max_batches=bleu_max_batches,
            max_samples=bleu_max_samples,
        )

        if scheduler is not None and scheduler_name == "plateau":
            scheduler.step(val_loss)

        if best_bleu is None or val_bleu > best_bleu:
            best_bleu = val_bleu
            save_checkpoint(config["save_path"], model, epoch, optimizer=optimizer, meta=config, exp=experiment)

        if avg_enabled:
            epoch_path = avg_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(str(epoch_path), model, epoch, optimizer=None, meta=config, exp=None)
            metric_value = float(val_bleu if avg_metric == "bleu" else val_loss)
            avg_records.append({"path": epoch_path.name, "metric_value": metric_value, "epoch": epoch})
            avg_records = sorted(
                avg_records,
                key=lambda rec: rec["metric_value"],
                reverse=(avg_metric == "bleu"),
            )
            for removed in avg_records[avg_num_last:]:
                (avg_dir / removed["path"]).unlink(missing_ok=True)
            avg_records = avg_records[:avg_num_last]
            manifest_payload = {
                "metric": avg_metric,
                "checkpoints": avg_records,
            }
            manifest_path.write_text(
                json.dumps(manifest_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if experiment is not None:
            experiment.log_text(f"Epoch {epoch}", step=epoch)
            experiment.log_text(f" train loss: {train_loss}", step=epoch)
            experiment.log_text(f" val loss: {val_loss}, val bleu: {val_bleu}", step=epoch)
            with experiment.context_manager("train"):
                experiment.log_metric("loss", train_loss, epoch=epoch)
            with experiment.context_manager("val"):
                experiment.log_metric("loss", val_loss, epoch=epoch)
                experiment.log_metric("bleu", val_bleu, epoch=epoch)

    return best_bleu

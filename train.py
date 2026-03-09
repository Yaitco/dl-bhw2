import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from sacrebleu.metrics import BLEU
from tqdm.auto import tqdm

from utils.utils import save_checkpoint, seq2seq_loss, trim_special_tokens


BATCH_STEP_SCHEDULERS = {"transformer_warmup", "transformerwarmup", "noamlr"}


def _normalize_name(name: str | None) -> str:
    return str(name or "").lower().replace("-", "_")


def _log_lr_batch(experiment, batch_log_state, lr_value: float, epoch: int):
    if experiment is None or batch_log_state is None:
        return
    batch_log_state["global_step"] += 1
    experiment.log_metric("lr_batch", float(lr_value), step=batch_log_state["global_step"], epoch=epoch)


def _next_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


def _build_reverse_prob_scheduler_cfg(dual_cfg: dict | None) -> dict:
    default_cfg = {"name": "linear", "start": 0.5, "end": 0.0}
    if dual_cfg is None:
        return default_cfg

    raw_cfg = dual_cfg.get("reverse_prob_scheduler")
    if raw_cfg is None:
        return default_cfg
    if isinstance(raw_cfg, (int, float)):
        value = float(raw_cfg)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"train.dual.reverse_prob_scheduler as number must be in [0,1]. Got: {value}")
        return {"name": "constant", "value": value}
    if not isinstance(raw_cfg, dict):
        raise ValueError("train.dual.reverse_prob_scheduler must be dict, number or null.")

    cfg = dict(raw_cfg)
    cfg["name"] = _normalize_name(cfg.get("name", "linear"))
    return cfg


def _reverse_batch_prob_for_epoch(epoch: int, total_epochs: int, scheduler_cfg: dict) -> float:
    name = _normalize_name(scheduler_cfg.get("name", "linear"))

    if name in {"linear", "lin"}:
        start = float(scheduler_cfg.get("start", 0.5))
        end = float(scheduler_cfg.get("end", 0.0))
        if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
            raise ValueError(f"reverse prob start/end must be in [0,1]. Got: start={start}, end={end}")
        progress = 1.0 if total_epochs <= 1 else (epoch - 1) / (total_epochs - 1)
        value = start + (end - start) * progress
    elif name in {"cosine", "cos"}:
        k = float(scheduler_cfg.get("k", 0.5))
        if k < 0.0:
            raise ValueError(f"reverse prob k must be >= 0 for cosine scheduler. Got: {k}")
        denom = max(int(total_epochs), 1)
        value = k * math.sin(math.pi * epoch / denom)
    elif name in {"constant", "const"}:
        value = float(scheduler_cfg.get("value", scheduler_cfg.get("start", 0.5)))
    else:
        raise ValueError(
            "train.dual.reverse_prob_scheduler.name must be 'linear', 'cosine' or 'constant'. "
            f"Got: {scheduler_cfg.get('name')}"
        )

    if not (0.0 <= value <= 1.0):
        raise ValueError(f"reverse batch probability must be in [0,1]. Got: {value}")
    return value


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
    direction: str = "straight",
    experiment=None,
    epoch: int | None = None,
    batch_log_state=None,
):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    scheduler_name = _normalize_name(scheduler_name)

    for src_ids, _, tgt_ids in tqdm(train_loader, desc=tqdm_desc):
        current_lr = optimizer.param_groups[0]["lr"]
        src_ids = src_ids.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)

        tgt_inp = tgt_ids[:, :-1]
        tgt_out = tgt_ids[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        logits = model(src_ids, tgt_inp, direction=direction)

        loss = seq2seq_loss(criterion, logits, tgt_out, pad_id)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None and scheduler_name in BATCH_STEP_SCHEDULERS:
            scheduler.step()

        _log_lr_batch(experiment, batch_log_state, current_lr, int(epoch or 0))

        tokens_in_batch = (tgt_out != pad_id).sum().item()
        total_loss += loss.item() * tokens_in_batch
        total_tokens += tokens_in_batch

    if scheduler is not None and scheduler_name not in {"plateau", *BATCH_STEP_SCHEDULERS}:
        scheduler.step()

    return total_loss / max(total_tokens, 1)


def training_epoch_dual(
    model,
    optimizer,
    criterion,
    forward_loader,
    reverse_loader,
    reverse_probability: float,
    tqdm_desc,
    device,
    pad_id,
    max_grad_norm: float = 1.0,
    scheduler=None,
    scheduler_name: str | None = None,
    experiment=None,
    epoch: int | None = None,
    batch_log_state=None,
):
    if len(forward_loader) == 0:
        raise ValueError("forward_loader is empty.")
    if len(reverse_loader) == 0:
        raise ValueError("reverse_loader is empty.")
    if not (0.0 <= reverse_probability <= 1.0):
        raise ValueError(f"reverse_probability must be in [0, 1]. Got: {reverse_probability}")

    model.train()
    total_loss = 0.0
    total_tokens = 0
    forward_steps = 0
    reverse_steps = 0
    scheduler_name = _normalize_name(scheduler_name)

    # Keep epoch length close to regular training.
    num_steps = len(forward_loader)
    forward_iter = iter(forward_loader)
    reverse_iter = iter(reverse_loader)

    for _ in tqdm(range(num_steps), desc=tqdm_desc):
        current_lr = optimizer.param_groups[0]["lr"]
        use_reverse = random.random() < reverse_probability
        direction = "reverse" if use_reverse else "straight"

        if use_reverse:
            (src_ids, _, tgt_ids), reverse_iter = _next_batch(reverse_loader, reverse_iter)
            reverse_steps += 1
        else:
            (src_ids, _, tgt_ids), forward_iter = _next_batch(forward_loader, forward_iter)
            forward_steps += 1

        src_ids = src_ids.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)

        tgt_inp = tgt_ids[:, :-1]
        tgt_out = tgt_ids[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        logits = model(src_ids, tgt_inp, direction=direction)

        loss = seq2seq_loss(criterion, logits, tgt_out, pad_id)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None and scheduler_name in BATCH_STEP_SCHEDULERS:
            scheduler.step()

        _log_lr_batch(experiment, batch_log_state, current_lr, int(epoch or 0))

        tokens_in_batch = (tgt_out != pad_id).sum().item()
        total_loss += loss.item() * tokens_in_batch
        total_tokens += tokens_in_batch

    if scheduler is not None and scheduler_name not in {"plateau", *BATCH_STEP_SCHEDULERS}:
        scheduler.step()

    return total_loss / max(total_tokens, 1), forward_steps, reverse_steps


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

        logits = model(src_ids, tgt_inp, direction="straight")
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
                direction="straight",
            )
            pred_tokens = trim_special_tokens(pred[0], bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)
            hypotheses.append(" ".join(tgt_vocab.decode(pred_tokens)))

            ref_tokens = trim_special_tokens(tgt_ids[i], bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)
            references.append(" ".join(tgt_vocab.decode(ref_tokens)))
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
    reverse_train_loader,
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
    train_cfg = config.get("train", {})
    scheduler_name = _normalize_name(config.get("scheduler", {}).get("name"))
    if scheduler is not None and not scheduler_name:
        scheduler_name = _normalize_name(type(scheduler).__name__)

    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError(f"train.label_smoothing must be in [0, 1). Got: {label_smoothing}")
    criterion_train = criterion
    if label_smoothing > 0.0:
        criterion_train = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)

    mode = _normalize_name(train_cfg.get("mode", "direct"))
    if mode in {"single", "forward", "straight"}:
        mode = "direct"
    if mode in {"double", "mixed", "bidirectional"}:
        mode = "dual"
    if mode not in {"direct", "dual"}:
        raise ValueError(f"train.mode must be 'direct' or 'dual'. Got: {train_cfg.get('mode')}")
    if mode == "dual" and reverse_train_loader is None:
        raise ValueError("reverse_train_loader is required when train.mode='dual'.")

    dual_cfg = train_cfg.get("dual", {}) if mode == "dual" else {}
    reverse_prob_scheduler_cfg = _build_reverse_prob_scheduler_cfg(dual_cfg) if mode == "dual" else None

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

    bleu_max_batches = train_cfg.get("bleu_max_batches")
    bleu_max_samples = train_cfg.get("bleu_max_samples")
    bleu_max_batches = None if bleu_max_batches is None else int(bleu_max_batches)
    bleu_max_samples = None if bleu_max_samples is None else int(bleu_max_samples)

    if avg_enabled:
        avg_dir.mkdir(parents=True, exist_ok=True)
        if avg_reset_on_start:
            for old_path in avg_dir.glob("epoch_*.pt"):
                old_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)

    batch_log_state = {"global_step": 0}
    if experiment is not None and mode == "dual":
        dual_scheduler_params = {
            f"dual_scheduler.{k}": v for k, v in reverse_prob_scheduler_cfg.items()
        }
        experiment.log_parameters(dual_scheduler_params)

    for epoch in range(1, int(num_epochs) + 1):
        reverse_prob = 0.0
        forward_steps = len(train_loader)
        reverse_steps = 0

        if mode == "dual":
            reverse_prob = _reverse_batch_prob_for_epoch(
                epoch=epoch,
                total_epochs=int(num_epochs),
                scheduler_cfg=reverse_prob_scheduler_cfg,
            )
            train_loss, forward_steps, reverse_steps = training_epoch_dual(
                model=model,
                optimizer=optimizer,
                criterion=criterion_train,
                forward_loader=train_loader,
                reverse_loader=reverse_train_loader,
                reverse_probability=reverse_prob,
                tqdm_desc=f"Training {epoch}/{num_epochs} [dual p_rev={reverse_prob:.3f}]",
                device=device,
                pad_id=pad_id,
                max_grad_norm=max_grad_norm,
                scheduler=scheduler,
                scheduler_name=scheduler_name,
                experiment=experiment,
                epoch=epoch,
                batch_log_state=batch_log_state,
            )
        else:
            train_loss = training_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion_train,
                train_loader=train_loader,
                tqdm_desc=f"Training {epoch}/{num_epochs} [direct]",
                device=device,
                pad_id=pad_id,
                max_grad_norm=max_grad_norm,
                scheduler=scheduler,
                scheduler_name=scheduler_name,
                direction="straight",
                experiment=experiment,
                epoch=epoch,
                batch_log_state=batch_log_state,
            )

        # Validation metric for model selection/checkpoint averaging is always de->en.
        val_loss = validation_epoch(
            model=model,
            criterion=criterion,
            val_loader=test_loader,
            tqdm_desc=f"Validating {epoch}/{num_epochs} (loss, de->en)",
            device=device,
            pad_id=pad_id,
        )
        val_bleu = validation_epoch_bleu(
            model=model,
            val_loader=test_loader,
            tqdm_desc=f"Validating {epoch}/{num_epochs} (bleu, de->en)",
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
            experiment.log_text(
                (
                    f" mode: {mode}, reverse_prob: {reverse_prob:.4f}, "
                    f"steps(de->en): {forward_steps}, steps(en->de): {reverse_steps}, "
                    f"scheduler: {scheduler_name or 'none'}, lr: {optimizer.param_groups[0]['lr']}, "
                    f"train loss: {train_loss}"
                ),
                step=epoch,
            )
            experiment.log_text(f" val loss (de->en): {val_loss}, val bleu (de->en): {val_bleu}", step=epoch)
            with experiment.context_manager("train"):
                experiment.log_metric("loss", train_loss, epoch=epoch)
                experiment.log_metric("lr", optimizer.param_groups[0]["lr"], epoch=epoch)
                experiment.log_metric("reverse_prob", reverse_prob, epoch=epoch)
                experiment.log_metric("dual_scheduler_reverse_prob", reverse_prob, epoch=epoch)
                experiment.log_metric("forward_steps", forward_steps, epoch=epoch)
                experiment.log_metric("reverse_steps", reverse_steps, epoch=epoch)
            with experiment.context_manager("val"):
                experiment.log_metric("loss", val_loss, epoch=epoch)
                experiment.log_metric("bleu", val_bleu, epoch=epoch)

    return best_bleu

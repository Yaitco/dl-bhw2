from sacrebleu.metrics import BLEU
import torch
import torch.nn as nn

from tqdm.auto import tqdm

from utils.utils import save_checkpoint, seq2seq_loss, trim_special_tokens


def corpus_bleu_sacrebleu(hypotheses, references):
    bleu = BLEU(tokenize="none")
    return float(bleu.corpus_score(hypotheses, [references]).score)


def training_epoch(model, optimizer, criterion, train_loader, tqdm_desc, device, pad_id, max_grad_norm: float = 1.0):
    model.train()
    total_loss = 0.0
    total_tokens = 0

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

        tokens_in_batch = (tgt_out != pad_id).sum().item()
        total_loss += loss.item() * tokens_in_batch
        total_tokens += tokens_in_batch

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
def validation_epoch_bleu(model, val_loader, tqdm_desc, device, pad_id, bos_id, eos_id, beam_size, max_new_tokens, tgt_vocab):
    model.eval()

    hypotheses = []
    references = []

    for src_ids, _, tgt_ids in tqdm(val_loader, desc=tqdm_desc):
        src_ids = src_ids.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)

        for i in range(src_ids.size(0)):
            pred = model.beam_search_decode(
                src_ids=src_ids[i].unsqueeze(0),
                bos_id=bos_id,
                eos_id=eos_id,
                max_new_tokens=max_new_tokens,
                beam_size=beam_size,
            )
            pred_tokens = trim_special_tokens(pred[0], bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)
            hyp = " ".join(tgt_vocab.decode(pred_tokens))
            hypotheses.append(hyp)

            ref_tokens = trim_special_tokens(tgt_ids[i], bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)
            ref = " ".join(tgt_vocab.decode(ref_tokens))
            references.append(ref)

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

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            tqdm_desc=f"Training {epoch}/{num_epochs}",
            device=device,
            pad_id=pad_id,
            max_grad_norm=max_grad_norm,
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
        )

        if scheduler is not None:
            if config["scheduler"]["name"] == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if best_bleu is None or val_bleu > best_bleu:
            best_bleu = val_bleu
            save_checkpoint(config["save_path"], model, epoch, optimizer=optimizer, meta=config, exp=experiment)

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

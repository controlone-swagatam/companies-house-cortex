"""
Training loop for CompaniesHouseModel.

Param groups follow the convention seen elsewhere in this codebase
(neo_cortex_model.py's get_param_groups): gate parameters get a much
higher learning rate than the rest of the model, since HardConcreteGate's
log_alpha needs to move fast relative to the encoder/embedding weights to
find a useful sparsity level within a reasonable number of steps.

Sparsity penalty: model.py's HardConcreteGate.l0_loss() returns a raw sum
over positions, not a rate — normalized here by seq_len and penalized
toward a target_sparsity via L1 distance, weighted by sparsity_lambda.
This is a fixed-weight penalty, not the two-gate dynamic lambda controller
described in the working doc's canonical reference (that's a two-gate
design; this model has one gate, matching model.py) — a fixed weight is
the simpler starting point, worth revisiting if training proves unstable.
"""
import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader

from model.companies_house_model import CompaniesHouseModel
from model.config import EmbeddingConfig
from model.dataset import CompanySequenceDataset, collate_fn, load_company_sequences
from model.vocab import EventVocab

logger = logging.getLogger("ch_pipeline.model.train")


def get_param_groups(model: CompaniesHouseModel, base_lr: float, gate_lr_multiplier: float = 50.0):
    gate_params = list(model.encoder.gate.parameters())
    gate_ids = {id(p) for p in gate_params}
    other_params = [p for p in model.parameters() if id(p) not in gate_ids]

    return [
        {"params": other_params, "lr": base_lr, "weight_decay": 1e-4},
        {"params": gate_params, "lr": base_lr * gate_lr_multiplier, "weight_decay": 0.0},
    ]


def sparsity_penalty(model: CompaniesHouseModel, seq_len: int, target_sparsity: float) -> torch.Tensor:
    raw_l0 = model.encoder.l0_loss()  # sum over up to max_seq_len positions
    # normalize by the actual gate length used (min of seq_len, gate's max_seq_len)
    gate_len = min(seq_len, model.encoder.gate.log_alpha.shape[0])
    open_rate = raw_l0 / gate_len
    return torch.abs(open_rate - target_sparsity)


@torch.no_grad()
def evaluate(model: CompaniesHouseModel, dataloader: DataLoader, device: str) -> dict:
    model.eval()
    total_category_correct = total_type_correct = total_subtype_correct = 0
    total_masked = 0
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.forward_training(
            batch["category_ids"], batch["type_ids"], batch["subtype_ids"],
            batch["company_ids"], batch["positions"], batch["attention_mask"],
        )
        total_loss += out["total_loss"].item()
        n_batches += 1

        mask_pos = out["mask_positions"]
        if mask_pos.any():
            cat_pred = out["category_logits"].argmax(dim=-1)
            type_pred = out["type_logits"].argmax(dim=-1)
            subtype_pred = out["subtype_logits"].argmax(dim=-1)

            total_category_correct += (cat_pred[mask_pos] == batch["category_ids"][mask_pos]).sum().item()
            total_type_correct += (type_pred[mask_pos] == batch["type_ids"][mask_pos]).sum().item()
            total_subtype_correct += (subtype_pred[mask_pos] == batch["subtype_ids"][mask_pos]).sum().item()
            total_masked += mask_pos.sum().item()

    model.train()
    return {
        "eval_loss": total_loss / max(n_batches, 1),
        "category_accuracy": total_category_correct / max(total_masked, 1),
        "type_accuracy": total_type_correct / max(total_masked, 1),
        "subtype_accuracy": total_subtype_correct / max(total_masked, 1),
        "n_masked_positions": total_masked,
    }


def train(
    train_paths: list[str],
    eval_paths: list[str],
    vocab_path: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    max_seq_len: int = 64,
    base_lr: float = 1e-3,
    target_sparsity: float = 0.15,
    sparsity_lambda: float = 1.0,
    device: str = "cpu",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    vocab = EventVocab.load(vocab_path)
    logger.info(
        "Vocab: category=%d type=%d subtype=%d company=%d",
        vocab.n_categories, vocab.n_types, vocab.n_subtypes, vocab.n_companies,
    )

    train_sequences = load_company_sequences(train_paths)
    eval_sequences = load_company_sequences(eval_paths) if eval_paths else {}
    logger.info("Train companies: %d, Eval companies: %d", len(train_sequences), len(eval_sequences))

    train_ds = CompanySequenceDataset(train_sequences, vocab, max_seq_len=max_seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    logger.info("Train sequences (companies with >=2 events): %d", len(train_ds))

    eval_dl = None
    if eval_sequences:
        eval_ds = CompanySequenceDataset(eval_sequences, vocab, max_seq_len=max_seq_len)
        eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        logger.info("Eval sequences: %d", len(eval_ds))

    cfg = EmbeddingConfig(max_seq_len=max_seq_len)
    model = CompaniesHouseModel(vocab, cfg, max_seq_len=max_seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d trainable params", n_params)

    optimizer = torch.optim.AdamW(get_param_groups(model, base_lr))

    for epoch in range(epochs):
        model.train()
        epoch_losses = {"total_loss": 0.0, "category_loss": 0.0, "type_loss": 0.0,
                         "subtype_loss": 0.0, "sparsity_loss": 0.0}
        n_batches = 0

        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.forward_training(
                batch["category_ids"], batch["type_ids"], batch["subtype_ids"],
                batch["company_ids"], batch["positions"], batch["attention_mask"],
            )
            sp_loss = sparsity_penalty(model, batch["category_ids"].shape[1], target_sparsity)
            loss = out["total_loss"] + sparsity_lambda * sp_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses["total_loss"] += out["total_loss"].item()
            epoch_losses["category_loss"] += out["category_loss"].item()
            epoch_losses["type_loss"] += out["type_loss"].item()
            epoch_losses["subtype_loss"] += out["subtype_loss"].item()
            epoch_losses["sparsity_loss"] += sp_loss.item()
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        logger.info(
            "Epoch %d/%d | loss=%.4f (cat=%.4f type=%.4f subtype=%.4f) sparsity=%.4f",
            epoch + 1, epochs, avg["total_loss"], avg["category_loss"],
            avg["type_loss"], avg["subtype_loss"], avg["sparsity_loss"],
        )

        if eval_dl is not None:
            eval_metrics = evaluate(model, eval_dl, device)
            logger.info(
                "  eval: loss=%.4f cat_acc=%.3f type_acc=%.3f subtype_acc=%.3f (n_masked=%d)",
                eval_metrics["eval_loss"], eval_metrics["category_accuracy"],
                eval_metrics["type_accuracy"], eval_metrics["subtype_accuracy"],
                eval_metrics["n_masked_positions"],
            )

        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_path": vocab_path,
            "config": cfg,
        }, checkpoint_path)
        logger.info("  checkpoint saved: %s", checkpoint_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train CompaniesHouseModel")
    parser.add_argument("--train-input", nargs="+", required=True, help="derived_events period_1 JSONL file(s)")
    parser.add_argument("--eval-input", nargs="*", default=[], help="derived_events period_2 JSONL file(s)")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-sparsity", type=float, default=0.15)
    parser.add_argument("--sparsity-lambda", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train(
        train_paths=args.train_input,
        eval_paths=args.eval_input,
        vocab_path=args.vocab,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        base_lr=args.lr,
        target_sparsity=args.target_sparsity,
        sparsity_lambda=args.sparsity_lambda,
        device=args.device,
    )


if __name__ == "__main__":
    main()

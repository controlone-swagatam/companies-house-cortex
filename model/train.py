"""
Two-phase training loop, per CORTEX_ARCHITECTURE.md §3.

Phase 1 (epochs 0 .. phase1_epochs-1):
    Gate 1 trains (dynamic lambda1, target 0.15). Gate 2 frozen at
    log_alpha=0.0 (~0.5 gate value, pass-through, no selection pressure).
Phase 2 (epochs phase1_epochs .. end):
    Gate 1 binarized (+5.0/-5.0 per position) and frozen at its Phase-1
    end state. Gate 2 unfrozen, trains (dynamic lambda2, target 0.05).

Loss = L_MLM + lambda_k * (L0_k/N - target_k)^2, where k is whichever
gate is currently active/trainable in the current phase — the frozen
gate's sparsity term is dropped (not just zero-weighted) since it has no
gradient to receive anyway.

Param groups: gate parameters get a much higher LR than the rest of the
model, matching the convention elsewhere in this codebase
(neo_cortex_model.py's get_param_groups) — HardConcreteGate's log_alpha
needs to move fast relative to encoder/embedding weights.
"""
import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader

from model.companies_house_model import CompaniesHouseModel
from model.config import EmbeddingConfig
from model.dataset import CompanySequenceDataset, collate_fn, load_company_sequences
from model.hstu import dynamic_lambda
from model.vocab import EventVocab

logger = logging.getLogger("ch_pipeline.model.train")


def get_param_groups(model: CompaniesHouseModel, base_lr: float, gate_lr_multiplier: float = 50.0):
    gate_params = list(model.encoder.gate1.parameters()) + list(model.encoder.gate2.parameters())
    gate_ids = {id(p) for p in gate_params}
    other_params = [p for p in model.parameters() if id(p) not in gate_ids]

    return [
        {"params": other_params, "lr": base_lr, "weight_decay": 1e-4},
        {"params": gate_params, "lr": base_lr * gate_lr_multiplier, "weight_decay": 0.0},
    ]


def phase_sparsity_loss(
    model: CompaniesHouseModel, seq_len: int, phase: int,
) -> torch.Tensor:
    """
    Returns the sparsity penalty for whichever gate is active in the
    current phase. Phase 1 -> gate1 (target from encoder.gate1_target),
    Phase 2 -> gate2. Dynamic lambda computed from that gate's
    deterministic (eval-style) values, per the canonical spec.
    """
    encoder = model.encoder
    gate = encoder.gate1 if phase == 1 else encoder.gate2
    target = encoder.gate1_target if phase == 1 else encoder.gate2_target

    gate_len = min(seq_len, gate.log_alpha.shape[0])
    det_values = gate.deterministic_gate_values(gate_len)
    lam = dynamic_lambda(det_values)

    l0_rate = encoder.l0_rate(gate, seq_len)
    return lam * (l0_rate - target) ** 2


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
    epochs: int = 50,
    phase1_epochs: int = 25,
    batch_size: int = 16,
    max_seq_len: int = 64,
    min_context: int = 1,
    base_lr: float = 1e-3,
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
    model = CompaniesHouseModel(vocab, cfg, max_seq_len=max_seq_len, min_context=min_context).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d trainable params", n_params)

    # Phase 1 setup: Gate 2 frozen at pass-through from the start.
    model.encoder.gate2.freeze_at_zero()
    phase = 1
    logger.info("Phase 1 started: Gate 1 training (target %.2f), Gate 2 frozen at pass-through",
                model.encoder.gate1_target)

    optimizer = torch.optim.AdamW(get_param_groups(model, base_lr))

    for epoch in range(epochs):
        if epoch == phase1_epochs and phase == 1:
            model.encoder.gate1.binarize_and_freeze()
            model.encoder.gate2.unfreeze()
            phase = 2
            # New optimizer: param groups changed (gate1 now has no grad,
            # gate2 does) — rebuild rather than silently carrying stale
            # momentum state for now-frozen gate1 params.
            optimizer = torch.optim.AdamW(get_param_groups(model, base_lr))
            logger.info("Phase 2 started: Gate 1 binarized+frozen, Gate 2 training (target %.2f)",
                        model.encoder.gate2_target)

        model.train()
        epoch_losses = {"total_loss": 0.0, "category_loss": 0.0, "type_loss": 0.0,
                         "subtype_loss": 0.0, "sparsity_loss": 0.0}
        n_batches = 0
        n_skipped_nonfinite = 0

        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.forward_training(
                batch["category_ids"], batch["type_ids"], batch["subtype_ids"],
                batch["company_ids"], batch["positions"], batch["attention_mask"],
            )
            sp_loss = phase_sparsity_loss(model, batch["category_ids"].shape[1], phase)
            loss = out["total_loss"] + sp_loss

            if not torch.isfinite(loss):
                n_skipped_nonfinite += 1
                logger.warning(
                    "Non-finite loss on a batch (epoch %d, phase %d) — skipping optimizer step. "
                    "total_loss=%s sparsity=%s",
                    epoch + 1, phase, out["total_loss"].item(), sp_loss.item(),
                )
                optimizer.zero_grad()
                continue

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
        active_gate_rate = model.encoder.l0_rate(
            model.encoder.gate1 if phase == 1 else model.encoder.gate2, max_seq_len
        ).item()
        logger.info(
            "Epoch %d/%d [phase %d] | loss=%.4f (cat=%.4f type=%.4f subtype=%.4f) "
            "sparsity=%.4f gate%d_rate=%.4f%s",
            epoch + 1, epochs, phase, avg["total_loss"], avg["category_loss"],
            avg["type_loss"], avg["subtype_loss"], avg["sparsity_loss"],
            phase, active_gate_rate,
            f" | skipped {n_skipped_nonfinite} non-finite batches" if n_skipped_nonfinite else "",
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
            "phase": phase,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_path": vocab_path,
            "config": cfg,
        }, checkpoint_path)
        logger.info("  checkpoint saved: %s", checkpoint_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train CompaniesHouseModel (two-phase)")
    parser.add_argument("--train-input", nargs="+", required=True)
    parser.add_argument("--eval-input", nargs="*", default=[])
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--phase1-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--min-context", type=int, default=1,
                         help="Min real predecessors before a position is maskable. "
                              "CH default is 1 (not GDELT's 10) — see companies_house_model.py")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train(
        train_paths=args.train_input,
        eval_paths=args.eval_input,
        vocab_path=args.vocab,
        output_dir=args.output_dir,
        epochs=args.epochs,
        phase1_epochs=args.phase1_epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        min_context=args.min_context,
        base_lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()

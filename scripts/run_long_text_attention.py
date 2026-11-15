"""Config-driven long-text masked-attention classifier."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import MaskedAttentionClassifier
from src.config import load_config
from src.datasets import build_masked_hidden_state_loaders
from src.loaders import load_hidden_states
from src.training import calculate_binary_metrics, evaluate_masked_loader


def resolve_device(config):
    configured_device = config["model"].get("device", "cpu")
    if str(configured_device).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(configured_device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config):
    experiment = config["long_text_attention"]
    training = config["training"]
    device = resolve_device(config)
    set_seed(experiment.get("seed", 123))

    results, correct = load_hidden_states(experiment["results_file"])
    generated_embeddings = results["generated_embeddings"]
    labels = np.asarray(correct, dtype=np.int64)
    if len(generated_embeddings) != len(labels):
        raise ValueError("The number of embedding sequences and labels must match.")
    if not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError("The dataset must contain both labels 0 and 1.")

    all_indices = np.arange(len(labels))
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=experiment.get("test_size", 0.2),
        random_state=experiment.get("seed", 123),
        stratify=labels,
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=experiment.get("validation_size", 0.2),
        random_state=experiment.get("seed", 123),
        stratify=labels[train_val_indices],
    )
    first_sample = torch.as_tensor(generated_embeddings[0])
    if first_sample.ndim != 2:
        raise ValueError("Each embedding must have shape [sequence_length, hidden_dim].")
    hidden_dim = int(first_sample.shape[-1])

    batch_size = min(experiment.get("micro_batch_size", 8), len(train_indices))
    train_loader, val_loader, test_loader = build_masked_hidden_state_loaders(
        generated_embeddings,
        labels,
        train_indices,
        val_indices,
        test_indices,
        max_seq_len=experiment.get("max_sequence_length"),
        batch_size=batch_size,
        device=device,
    )
    model_config = {
        "hidden_dim": hidden_dim,
        "attention_dim": experiment.get("attention_dim", 128),
        "reducer_dropout": experiment.get("reducer_dropout", 0.1),
        "mlp_dropout": experiment.get("mlp_dropout", 0.25),
    }
    model = MaskedAttentionClassifier(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    train_iterator = repeat_loader(train_loader)
    best_val_loss = float("inf")
    evaluations_without_improvement = 0
    checkpoint_path = Path(experiment["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for step in tqdm(range(1, experiment.get("steps", 1001) + 1), desc="Training"):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulation_steps = experiment.get("gradient_accumulation_steps", 16)
        for _ in range(accumulation_steps):
            hidden_states, token_mask, batch_labels = next(train_iterator)
            hidden_states = hidden_states.to(device, non_blocking=True)
            token_mask = token_mask.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = criterion(model(hidden_states, token_mask), batch_labels)
                normalized_loss = loss / accumulation_steps
            scaler.scale(normalized_loss).backward()
            accumulated_loss += loss.detach().item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), experiment.get("max_grad_norm", 1.0)
        )
        scaler.step(optimizer)
        scaler.update()

        evaluate_every = experiment.get("evaluate_every", 50)
        if step % evaluate_every and step != experiment.get("steps", 1001):
            continue
        validation = evaluate_masked_loader(
            model, val_loader, criterion, device=device, amp_dtype=amp_dtype
        )
        val_metrics = calculate_binary_metrics(
            validation["y_true"],
            validation["y_score"],
            experiment.get("decision_threshold", 0.5),
        )
        tqdm.write(
            f"Step {step} | Train loss: {accumulated_loss / accumulation_steps:.4f} | "
            f"Val loss: {validation['loss']:.4f} | Val AUROC: {val_metrics['roc_auc']:.4f}"
        )
        if validation["loss"] < best_val_loss - experiment.get("min_delta", 1e-4):
            best_val_loss = validation["loss"]
            evaluations_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "max_seq_len": experiment.get("max_sequence_length"),
                    "decision_threshold": experiment.get("decision_threshold", 0.5),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "validation_metrics": val_metrics,
                    "seed": experiment.get("seed", 123),
                    "train_indices": train_indices.tolist(),
                    "val_indices": val_indices.tolist(),
                    "test_indices": test_indices.tolist(),
                },
                checkpoint_path,
            )
        else:
            evaluations_without_improvement += 1
            if evaluations_without_improvement >= experiment.get("early_stopping_patience", 6):
                tqdm.write(f"Early stopping at step {step}.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_results = evaluate_masked_loader(
        model, test_loader, criterion, device=device, amp_dtype=amp_dtype
    )
    test_metrics = calculate_binary_metrics(
        test_results["y_true"],
        test_results["y_score"],
        checkpoint["decision_threshold"],
    )
    result_path = Path(experiment["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        file.write("Test results - positive class = label 1\n")
        file.write(f"Best training step: {checkpoint['step']}\n")
        file.write(f"Test loss: {test_results['loss']:.4f}\n")
        for name, value in test_metrics.items():
            file.write(f"{name}: {value:.4f}\n")
        file.write(f"Checkpoint: {checkpoint_path}\n")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Test metrics: {test_metrics}")
    return model, test_metrics


def repeat_loader(data_loader):
    while True:
        yield from data_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()

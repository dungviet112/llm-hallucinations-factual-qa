"""Training and evaluation helpers shared by classifier entry points."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def pool_generated_embeddings(generated_embeddings):
    """Mean-pool each token-level embedding sequence."""
    return torch.stack([
        torch.as_tensor(sequence, dtype=torch.float32).mean(dim=0)
        for sequence in generated_embeddings
    ])


def train_classifier(
    inputs,
    labels,
    classifier,
    device="cpu",
    batch_size=128,
    learning_rate=1e-4,
    weight_decay=1e-2,
    steps=1001,
    test_size=0.2,
    random_state=123,
    save_path=None,
):
    """Train a fixed-size classifier and return metrics and the model."""
    x_train, x_test, y_train, y_test = train_test_split(
        np.asarray(inputs), np.asarray(labels).astype(int),
        test_size=test_size, random_state=random_state,
    )
    model = classifier(x_train.shape[1]).to(device)
    x_train = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    x_test = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_train, dtype=torch.long, device=device)
    y_test = torch.as_tensor(y_test, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    model.train()
    for _ in tqdm(range(steps), desc="Training"):
        optimizer.zero_grad()
        sample = torch.randperm(x_train.shape[0], device=device)[:batch_size]
        loss = torch.nn.functional.cross_entropy(model(x_train[sample]), y_train[sample])
        loss.backward()
        optimizer.step()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return model, evaluate_classifier(model, x_test, y_test)


def evaluate_classifier(model, inputs, labels):
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs), dim=1)[:, 1].detach().cpu().numpy()
    true_labels = labels.detach().cpu().numpy()
    predictions = (probabilities >= 0.5).astype(int)
    false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, probabilities)
    valid = np.flatnonzero(false_positive_rate <= 0.05 + 1e-12)
    print("Evaluating...")
    metrics = {
        "roc_auc": roc_auc_score(true_labels, probabilities),
        "accuracy": accuracy_score(true_labels, predictions),
        "f1": f1_score(true_labels, predictions, zero_division=0),
        "tpr_at_5_fpr": float(true_positive_rate[valid].max()) if len(valid) else 0.0,
    }
    return metrics


def evaluate_loader(model, data_loader, device="cpu"):
    """Evaluate a classifier whose variable-length inputs come from a loader."""
    model.eval()
    probabilities = []
    labels = []
    with torch.inference_mode():
        for inputs, batch_labels in data_loader:
            logits = model(inputs.to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
            labels.append(batch_labels.cpu())
    true_labels = torch.cat(labels).numpy()
    scores = torch.cat(probabilities).numpy()
    predictions = (scores >= 0.5).astype(int)
    false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, scores)
    valid = np.flatnonzero(false_positive_rate <= 0.05 + 1e-12)
    return {
        "roc_auc": roc_auc_score(true_labels, scores),
        "accuracy": accuracy_score(true_labels, predictions),
        "f1": f1_score(true_labels, predictions, zero_division=0),
        "tpr_at_5_fpr": float(true_positive_rate[valid].max()) if len(valid) else 0.0,
    }


def evaluate_masked_loader(model, data_loader, criterion, device="cpu", amp_dtype=None):
    """Evaluate a masked-attention model and retain scores for threshold metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    probabilities = []
    labels = []
    use_amp = device.type == "cuda" if isinstance(device, torch.device) else str(device).startswith("cuda")
    with torch.inference_mode():
        for hidden_states, token_mask, batch_labels in data_loader:
            hidden_states = hidden_states.to(device, non_blocking=True)
            token_mask = token_mask.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=amp_dtype or torch.float16,
                enabled=use_amp,
            ):
                logits = model(hidden_states, token_mask)
                loss = criterion(logits, batch_labels)
            batch_size = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            probabilities.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
            labels.append(batch_labels.cpu())
    return {
        "loss": total_loss / total_samples,
        "y_true": torch.cat(labels).numpy().astype(np.int64),
        "y_score": torch.cat(probabilities).numpy(),
    }


def calculate_binary_metrics(y_true, y_score, threshold=0.5):
    """Calculate AUROC, average precision, accuracy, F1, and TPR@5%FPR."""
    predictions = (y_score >= threshold).astype(np.int64)
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true, y_score, pos_label=1, drop_intermediate=False
    )
    valid = np.flatnonzero(false_positive_rate <= 0.05 + 1e-12)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "accuracy": float((predictions == y_true).mean()),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tpr_at_5_fpr": float(true_positive_rate[valid].max()),
    }
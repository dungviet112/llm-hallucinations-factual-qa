import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from utils import load_hidden_states
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import SingleMLP_Classifier


# =========================================================
# Configuration
# =========================================================

gpu = "0"
device = torch.device(
    f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

MAX_SEQ_LEN = 2048
REDUCED_LEN_1 = 512
REDUCED_LEN_2 = 128

# Use gradient accumulation to maintain an effective batch size of 128.
micro_batch_size = 8
gradient_accumulation_steps = 16
effective_batch_size = (
    micro_batch_size * gradient_accumulation_steps
)

num_training_steps = 1001
learning_rate = 1e-4
weight_decay = 1e-2
reducer_dropout = 0.2
mlp_dropout = 0.25

save_model_dir = "./models/Llama-3.1-8B-Instruct"
save_name = "single_mlp_context_emb_FAVA_reduced_dim.pt"
os.makedirs(save_model_dir, exist_ok=True)

print(f"Effective batch size: {effective_batch_size}")


# =========================================================
# Dataset
# =========================================================

class HiddenStateDataset(Dataset):
    def __init__(self, hidden_states, labels, indices):
        self.hidden_states = hidden_states
        self.labels = labels
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]

        # Shape: [sequence_length, hidden_dim]
        hidden_state = torch.as_tensor(
            self.hidden_states[original_index],
            dtype=torch.float32,
        )

        if hidden_state.ndim != 2:
            raise ValueError(
                "Each hidden-state sample must have shape "
                "[sequence_length, hidden_dim]. "
                f"Received {hidden_state.shape}."
            )

        label = int(self.labels[original_index])

        return hidden_state, label


def collate_hidden_states(batch, max_seq_len):
    """
    Right-pads short sequences with zero hidden states and truncates
    sequences longer than max_seq_len.

    Output:
        hidden_states: [batch_size, max_seq_len, hidden_dim]
        labels:        [batch_size]
    """
    sequences, labels = zip(*batch)

    hidden_dim = sequences[0].shape[-1]

    padded_hidden_states = torch.zeros(
        len(sequences),
        max_seq_len,
        hidden_dim,
        dtype=torch.float32,
    )

    for index, sequence in enumerate(sequences):
        if sequence.shape[-1] != hidden_dim:
            raise ValueError(
                "All samples must have the same hidden dimension."
            )

        sequence_length = min(
            sequence.shape[0],
            max_seq_len,
        )

        padded_hidden_states[index, :sequence_length] = (
            sequence[:sequence_length]
        )

    labels = torch.tensor(labels, dtype=torch.long)

    return padded_hidden_states, labels


# =========================================================
# Sequence-length reducer
# =========================================================

class SequenceLengthReducer(nn.Module):
    """
    Reduces only the sequence dimension:

        [B, 2048, D]
        -> [B, 512, D]
        -> [B, 128, D]
        -> [B, 1, D]
        -> [B, D]

    The LLM hidden dimension D remains unchanged.
    """

    def __init__(
        self,
        max_seq_len,
        reduced_len_1,
        reduced_len_2,
        dropout=0.2,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.reducer = nn.Sequential(
            nn.Linear(max_seq_len, reduced_len_1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(reduced_len_1, reduced_len_2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(reduced_len_2, 1),
        )

    def forward(self, hidden_states):
        # hidden_states: [B, T, D]

        if hidden_states.ndim != 3:
            raise ValueError(
                "Expected hidden states with shape "
                "[batch_size, sequence_length, hidden_dim]."
            )

        if hidden_states.shape[1] != self.max_seq_len:
            raise ValueError(
                f"Expected sequence length {self.max_seq_len}, "
                f"received {hidden_states.shape[1]}."
            )

        # [B, T, D] -> [B, D, T]
        hidden_states = hidden_states.transpose(1, 2)

        # [B, D, 2048] -> [B, D, 512]
        # -> [B, D, 128] -> [B, D, 1]
        reduced_hidden_states = self.reducer(hidden_states)

        # [B, D, 1] -> [B, D]
        return reduced_hidden_states.squeeze(-1)


# =========================================================
# Combined model
# =========================================================

class HallucinationClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim,
        max_seq_len,
        reduced_len_1,
        reduced_len_2,
        reducer_dropout=0.2,
        mlp_dropout=0.2,
    ):
        super().__init__()

        self.sequence_reducer = SequenceLengthReducer(
            max_seq_len=max_seq_len,
            reduced_len_1=reduced_len_1,
            reduced_len_2=reduced_len_2,
            dropout=reducer_dropout,
        )

        # Single MLP classifier backbone
        self.classifier = SingleMLP_Classifier(
            input_shape=hidden_dim,
            dropout=mlp_dropout,
        )

    def forward(self, hidden_states):
        # [B, 2048, D] -> [B, D]
        fixed_size_vector = self.sequence_reducer(hidden_states)

        # [B, D] -> [B, 2]
        logits = self.classifier(fixed_size_vector)

        return logits


# =========================================================
# Load and split data
# =========================================================
results_file = "./results/460_samples_FAVA_Llama3.1-8B/Llama-3.1-8B-Instruct_fava_annotation_start-0_end-10000_10_25.pickle"
results, correct = load_hidden_states(results_file)

generated_embeddings = results["generated_embeddings"]
labels = np.asarray(correct, dtype=np.int64)

if len(generated_embeddings) != len(labels):
    raise ValueError(
        "The number of embedding sequences and labels must match."
    )

all_indices = np.arange(len(labels))

train_indices, test_indices = train_test_split(
    all_indices,
    test_size=0.2,
    random_state=123,
    stratify=labels,
)

# Determine the unchanged LLM hidden dimension.
first_sample = torch.as_tensor(generated_embeddings[0])

if first_sample.ndim != 2:
    raise ValueError(
        "Expected each sample to have shape "
        "[sequence_length, hidden_dim]."
    )

hidden_dim = first_sample.shape[-1]

print(f"Number of training samples: {len(train_indices)}")
print(f"Number of test samples: {len(test_indices)}")
print(f"LLM hidden dimension: {hidden_dim}")


# =========================================================
# DataLoaders
# =========================================================

train_dataset = HiddenStateDataset(
    hidden_states=generated_embeddings,
    labels=labels,
    indices=train_indices,
)

test_dataset = HiddenStateDataset(
    hidden_states=generated_embeddings,
    labels=labels,
    indices=test_indices,
)

collate_function = partial(
    collate_hidden_states,
    max_seq_len=MAX_SEQ_LEN,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=micro_batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=device.type == "cuda",
    collate_fn=collate_function,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=micro_batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=device.type == "cuda",
    collate_fn=collate_function,
)


# Create a non-caching infinite iterator for 1001 training steps.
def repeat_loader(data_loader):
    while True:
        yield from data_loader


train_iterator = iter(repeat_loader(train_loader))


# =========================================================
# Initialize model
# =========================================================

classifier_model = HallucinationClassifier(
    hidden_dim=hidden_dim,
    max_seq_len=MAX_SEQ_LEN,
    reduced_len_1=REDUCED_LEN_1,
    reduced_len_2=REDUCED_LEN_2,
    reducer_dropout=reducer_dropout,
    mlp_dropout=mlp_dropout,
).to(device)

optimizer = torch.optim.AdamW(
    classifier_model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

criterion = nn.CrossEntropyLoss()

# Mixed precision reduces GPU memory usage.
use_amp = device.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


# =========================================================
# Jointly train the reducer and MLP
# =========================================================

classifier_model.train()

progress_bar = tqdm(range(num_training_steps))

for _ in progress_bar:
    optimizer.zero_grad(set_to_none=True)

    accumulated_loss = 0.0

    for _ in range(gradient_accumulation_steps):
        hidden_states, batch_labels = next(train_iterator)

        hidden_states = hidden_states.to(
            device,
            non_blocking=True,
        )
        batch_labels = batch_labels.to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(enabled=use_amp):
            predictions = classifier_model(hidden_states)
            loss = criterion(predictions, batch_labels)

            # Normalize the gradients over accumulated batches
            normalized_loss = (
                loss / gradient_accumulation_steps
            )

        scaler.scale(normalized_loss).backward()
        accumulated_loss += loss.detach().item()

    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(
        classifier_model.parameters(),
        max_norm=1.0,
    )

    scaler.step(optimizer)
    scaler.update()

    progress_bar.set_postfix(
        loss=(
            accumulated_loss /
            gradient_accumulation_steps
        )
    )


# =========================================================
# Evaluation
# =========================================================

classifier_model.eval()

test_probabilities = []
test_labels = []

with torch.inference_mode():
    for hidden_states, batch_labels in test_loader:
        hidden_states = hidden_states.to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = classifier_model(hidden_states)

        probabilities = torch.softmax(
            logits.float(),
            dim=1,
        )[:, 1]

        test_probabilities.append(probabilities.cpu())
        test_labels.append(batch_labels)

test_probabilities = torch.cat(test_probabilities)
test_labels = torch.cat(test_labels)

y_true = test_labels.numpy().astype(np.int64)
y_score = test_probabilities.numpy()

# Binary predictions using the standard 0.5 threshold.
decision_threshold = 0.5
y_pred = (y_score >= decision_threshold).astype(np.int64)

context_emb_roc = roc_auc_score(y_true, y_score)
context_emb_acc = (y_pred == y_true).mean()

# F1 for positive class 1 at threshold 0.5.
context_emb_f1 = f1_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0,
)

# Compute the empirical ROC curve.
fpr, tpr, roc_thresholds = roc_curve(
    y_true,
    y_score,
    pos_label=1,
    drop_intermediate=False,
)

# Highest TPR whose empirical FPR does not exceed 5%.
target_fpr = 0.05
valid_indices = np.flatnonzero(fpr <= target_fpr + 1e-12)

best_index = valid_indices[
    np.argmax(tpr[valid_indices])
]

tpr_at_5_fpr = tpr[best_index]

print(
    f"Contextual Embeddings\n"
    f"ROC AUC: {context_emb_roc:.4f}\n"
    f"Accuracy@0.5: {context_emb_acc:.4f}\n"
    f"F1@0.5: {context_emb_f1:.4f}\n"
    f"TPR@5%FPR: {tpr_at_5_fpr:.4f}\n"
)


# =========================================================
# Save reducer and classifier together
# =========================================================

torch.save(
    {
        "model_state_dict": classifier_model.state_dict(),
        "hidden_dim": hidden_dim,
        "max_seq_len": MAX_SEQ_LEN,
        "reduced_len_1": REDUCED_LEN_1,
        "reduced_len_2": REDUCED_LEN_2,
    },
    os.path.join(save_model_dir, save_name),
)
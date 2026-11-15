"""Dataset classes and batching utilities for hidden-state experiments."""

from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class HiddenStateDataset(Dataset):
    def __init__(self, hidden_states, labels, indices=None):
        self.hidden_states = hidden_states
        self.labels = np.asarray(labels)
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        hidden_state = torch.as_tensor(
            self.hidden_states[original_index],
            dtype=torch.float32,
        )
        if hidden_state.ndim != 2:
            raise ValueError(
                "Each hidden-state sample must have shape "
                f"[sequence_length, hidden_dim], received {tuple(hidden_state.shape)}."
            )
        return hidden_state, int(self.labels[original_index])


def collate_hidden_states(batch, max_seq_len):
    """Right-pad or truncate hidden-state sequences to a fixed length."""
    sequences, labels = zip(*batch)
    hidden_dim = sequences[0].shape[-1]
    padded = torch.zeros(
        len(sequences), max_seq_len, hidden_dim, dtype=torch.float32
    )
    for index, sequence in enumerate(sequences):
        if sequence.shape[-1] != hidden_dim:
            raise ValueError("All samples must have the same hidden dimension.")
        sequence_length = min(sequence.shape[0], max_seq_len)
        padded[index, :sequence_length] = sequence[:sequence_length]
    return padded, torch.tensor(labels, dtype=torch.long)


def build_hidden_state_loaders(
    hidden_states,
    labels,
    train_indices,
    test_indices,
    max_seq_len,
    batch_size,
    device="cpu",
):
    """Build train and test loaders using the long-text batching policy."""
    collate_fn = partial(collate_hidden_states, max_seq_len=max_seq_len)
    common = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": str(device).startswith("cuda"),
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(
        HiddenStateDataset(hidden_states, labels, train_indices),
        shuffle=True,
        **common,
    )
    test_loader = DataLoader(
        HiddenStateDataset(hidden_states, labels, test_indices),
        shuffle=False,
        **common,
    )
    return train_loader, test_loader


def collate_masked_hidden_states(batch, max_seq_len=None):
    """Dynamically pad hidden states and return a valid-token mask."""
    sequences, labels = zip(*batch)
    if max_seq_len is not None and max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive or None.")
    hidden_dim = sequences[0].shape[-1]
    lengths = [
        sequence.shape[0] if max_seq_len is None else min(sequence.shape[0], max_seq_len)
        for sequence in sequences
    ]
    if any(length == 0 for length in lengths):
        raise ValueError("Every sample must contain at least one token.")
    batch_max_len = max(lengths)
    padded = torch.zeros(len(sequences), batch_max_len, hidden_dim, dtype=torch.float32)
    token_mask = torch.zeros(len(sequences), batch_max_len, dtype=torch.bool)
    for index, (sequence, length) in enumerate(zip(sequences, lengths)):
        if sequence.shape[-1] != hidden_dim:
            raise ValueError("All samples must have the same hidden dimension.")
        padded[index, :length] = sequence[:length]
        token_mask[index, :length] = True
    return padded, token_mask, torch.tensor(labels, dtype=torch.long)


def build_masked_hidden_state_loaders(
    hidden_states,
    labels,
    train_indices,
    val_indices,
    test_indices,
    max_seq_len,
    batch_size,
    device="cpu",
):
    """Build train, validation, and test loaders for masked attention."""
    collate_fn = partial(collate_masked_hidden_states, max_seq_len=max_seq_len)
    common = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": str(device).startswith("cuda"),
        "collate_fn": collate_fn,
    }
    return tuple(
        DataLoader(
            HiddenStateDataset(hidden_states, labels, indices),
            shuffle=shuffle,
            drop_last=drop_last,
            **common,
        )
        for indices, shuffle, drop_last in (
            (train_indices, True, True),
            (val_indices, False, False),
            (test_indices, False, False),
        )
    )
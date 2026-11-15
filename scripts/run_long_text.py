"""Config-driven long-text classifier training entry point."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import HallucinationClassifier
from src.config import load_config
from src.datasets import build_hidden_state_loaders
from src.loaders import load_hidden_states
from src.training import evaluate_loader


def repeat_loader(data_loader):
	while True:
		yield from data_loader


def resolve_device(config):
	configured_device = config["model"].get("device", "cpu")
	if str(configured_device).startswith("cuda") and not torch.cuda.is_available():
		return torch.device("cpu")
	return torch.device(configured_device)


def train_long_text(config):
	device = resolve_device(config)
	dataset_config = config["dataset"]
	training_config = config["training"]
	long_text_config = config["long_text"]
	paths_config = config["paths"]

	results, correct = load_hidden_states(dataset_config["results_file"])
	generated_embeddings = results["generated_embeddings"]
	labels = np.asarray(correct, dtype=np.int64)
	if len(generated_embeddings) != len(labels):
		raise ValueError("The number of embedding sequences and labels must match.")

	indices = np.arange(len(labels))
	train_indices, test_indices = train_test_split(
		indices,
		test_size=training_config.get("test_size", 0.2),
		random_state=training_config.get("random_state", 123),
		stratify=labels,
	)
	first_sample = torch.as_tensor(generated_embeddings[0])
	if first_sample.ndim != 2:
		raise ValueError("Expected each sample to have shape [sequence_length, hidden_dim].")
	hidden_dim = first_sample.shape[-1]

	micro_batch_size = long_text_config.get("micro_batch_size", 8)
	accumulation_steps = long_text_config.get("gradient_accumulation_steps", 16)
	train_loader, test_loader = build_hidden_state_loaders(
		generated_embeddings,
		labels,
		train_indices,
		test_indices,
		max_seq_len=long_text_config["max_sequence_length"],
		batch_size=micro_batch_size,
		device=device,
	)

	model = HallucinationClassifier(
		hidden_dim=hidden_dim,
		max_seq_len=long_text_config["max_sequence_length"],
		reduced_len_1=long_text_config["reduced_length_1"],
		reduced_len_2=long_text_config["reduced_length_2"],
		reducer_dropout=long_text_config["reducer_dropout"],
		mlp_dropout=long_text_config["mlp_dropout"],
	).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=training_config["learning_rate"],
		weight_decay=training_config["weight_decay"],
	)
	criterion = torch.nn.CrossEntropyLoss()
	use_amp = device.type == "cuda"
	scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
	train_iterator = repeat_loader(train_loader)
	model.train()

	for _ in tqdm(range(training_config["steps"]), desc="Training"):
		optimizer.zero_grad(set_to_none=True)
		for _ in range(accumulation_steps):
			hidden_states, batch_labels = next(train_iterator)
			hidden_states = hidden_states.to(device, non_blocking=True)
			batch_labels = batch_labels.to(device, non_blocking=True)
			with torch.cuda.amp.autocast(enabled=use_amp):
				loss = criterion(model(hidden_states), batch_labels)
				normalized_loss = loss / accumulation_steps
			scaler.scale(normalized_loss).backward()
		scaler.unscale_(optimizer)
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		scaler.step(optimizer)
		scaler.update()

	metrics = evaluate_loader(model, test_loader, device=device)
	result_path = Path(
		long_text_config.get(
			"result_file",
			Path(paths_config["result_dir"]) / "long_text_scores.txt",
		)
	)
	result_path.parent.mkdir(parents=True, exist_ok=True)
	with result_path.open("a", encoding="utf-8") as file:
		for name, value in metrics.items():
			file.write(f"{name}: {value:.4f}\n")

	model_path = Path(paths_config["model_dir"]) / long_text_config.get(
		"save_name", "long_text_classifier.pt"
	)
	model_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"hidden_dim": hidden_dim,
			"max_seq_len": long_text_config["max_sequence_length"],
			"reduced_len_1": long_text_config["reduced_length_1"],
			"reduced_len_2": long_text_config["reduced_length_2"],
		},
		model_path,
	)
	print(f"Device: {device}")
	print(f"Saved model to {model_path}")
	print(f"Metrics: {metrics}")
	return model, metrics


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="config/experiment.yaml")
	args = parser.parse_args()
	train_long_text(load_config(args.config))


if __name__ == "__main__":
	main()

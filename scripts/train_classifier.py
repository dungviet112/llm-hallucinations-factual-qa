"""Train a fixed-size classifier using config/experiment.yaml."""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.classifier import (
    DNN_Classifier,
    RNN_Classifier,
    SingleMLP_Classifier,
    Transformer_Residual_Classifier,
)
from src.config import load_config
from src.loaders import load_hidden_states
from src.training import pool_generated_embeddings, train_classifier


ARCHITECTURES = {
    "single_mlp": SingleMLP_Classifier,
    "dnn": DNN_Classifier,
    "transformer_residual": Transformer_Residual_Classifier,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    results, labels = load_hidden_states(config["dataset"]["results_file"])
    inputs = pool_generated_embeddings(results["generated_embeddings"])
    architecture_name = config["classifier"]["architecture"]
    if architecture_name not in ARCHITECTURES:
        raise ValueError(f"Unsupported fixed-size architecture: {architecture_name}")
    training = config["training"]
    paths = config["paths"]
    model_name = Path(config["model"]["name"]).name
    save_model_dir = Path(paths["model_dir"]) / model_name
    Path(save_model_dir).mkdir(parents=True, exist_ok=True)
    save_path = save_model_dir / f"{architecture_name}_{config['dataset']['name']}.pt"
    device = config["model"].get("device", "cpu")
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        device = "cpu"
    print(f"Training {architecture_name} classifier on {device}...")
    _, metrics = train_classifier(
        inputs.numpy(),
        np.asarray(labels),
        ARCHITECTURES[architecture_name],
        device=device,
        batch_size=training["batch_size"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        steps=training["steps"],
        test_size=training["test_size"],
        random_state=training["random_state"],
        save_path=save_path,
    )
    print(metrics)


if __name__ == "__main__":
    main()
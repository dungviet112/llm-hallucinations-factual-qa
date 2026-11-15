"""Run inference with the shared package and YAML configuration."""

import argparse
import json

import torch

from src.config import load_config
from src.inference import infer_hallucination, load_inference_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    configured_device = config["model"].get("device", "cpu")
    device = configured_device if torch.cuda.is_available() else "cpu"
    model_config = config["model"]
    tokenizer, model, classifier = load_inference_models(
        model_config["name"],
        model_config["cache_dir"],
        args.checkpoint,
        device=device,
        feature=config["classifier"].get("feature", "context_emb"),
    )
    result = infer_hallucination(
        args.question,
        tokenizer,
        model,
        classifier,
        feature=config["classifier"].get("feature", "context_emb"),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
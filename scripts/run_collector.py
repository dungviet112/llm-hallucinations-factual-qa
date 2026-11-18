import argparse
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts import answer_trivia, get_integrated_gradients
from src.collection import ActivationStore, answer_fava, resolve_model_spec
from src.config import load_config
from src.loaders import load_dataset, load_llm
from src.proverqa import collect_proverqa, get_prompt_config


def collect_fava(dataset, model, tokenizer, model_name, device):
	embeddings, labels = answer_fava(
		model_name=model_name,
		data=dataset,
		model=model,
		tokenizer=tokenizer,
		device=device,
	)
	results = defaultdict(list)
	results["generated_embeddings"] = embeddings
	results["correct"] = labels
	return results


def collect_trivia(
	dataset,
	model,
	tokenizer,
	model_name,
	spec,
	collection_config,
):
	activation_store = ActivationStore()
	activation_store.register(model, spec)
	results = defaultdict(list)
	try:
		for question, answers in tqdm(dataset):
			activation_store.clear()
			response, text, logits, start_pos, correct, embeddings = answer_trivia(
				question,
				answers,
				model,
				tokenizer,
				model_name,
			)
			first_fc, first_attention = activation_store.collect(start_pos, spec)
			attributes = get_integrated_gradients(
				question,
				model,
				tokenizer,
				model_name,
				steps=collection_config.get("integrated_gradients_steps", 64),
				batch_size=collection_config.get("integrated_gradients_batch_size", 4),
			)
			results["question"].append(question)
			results["answers"].append(answers)
			results["response"].append(response)
			results["str_response"].append(text)
			results["logits"].append(logits.float().cpu().numpy())
			results["start_pos"].append(start_pos)
			results["correct"].append(correct)
			results["first_fully_connected"].append(first_fc)
			results["first_attention"].append(first_attention)
			results["attributes_first"].append(attributes)
			results["generated_embeddings"].append(np.stack(embeddings))
	finally:
		activation_store.close()
	return results


def save_results(results, output_dir, model_name, dataset_name):
	output_dir.mkdir(parents=True, exist_ok=True)
	model_key = model_name.rstrip("/").split("/")[-1]
	output_path = output_dir / (
		f"{model_key}_{dataset_name}_{datetime.now().month}_{datetime.now().day}.pickle"
	)
	with output_path.open("wb") as file:
		pickle.dump(results, file)
	return output_path


def collect_prover_qa(config, model, tokenizer):
	"""Run the ProverQA collector using explicit YAML configuration."""
	dataset_config = config["dataset"]
	# prover_config = config.get("proverqa", {})
	reasoning = config["model"].get("reasoning", False)
	print(f"Collecting ProverQA with reasoning={reasoning}")
	system_prompt, prompt_template, max_new_tokens = get_prompt_config(
		reasoning=reasoning,
		# system_prompt=prover_config.get("system_prompt"),
		# prompt_template=prover_config.get("prompt_template"),
	)
	json_path = Path(dataset_config["data_dir"]) / "provergen-5000.json"
	model_key = config["model"]["name"].rstrip("/").split("/")[-1]
	save_dir = Path(config["paths"]["result_dir"]) / f"{dataset_config['name']}_{model_key}"

	return collect_proverqa(
		model=model,
		tokenizer=tokenizer,
		json_path=json_path,
		save_dir=save_dir,
		system_prompt=system_prompt,
		prompt_template=prompt_template,
		max_new_tokens=max_new_tokens,
	)


def run(config):
	model_config = config["model"]
	dataset_config = config["dataset"]
	collection_config = config.get("collection", {})
	dataset_name = dataset_config["name"]
	model_name = model_config["name"]
	layer_number = collection_config.get("layer_number", -1)

	if torch.cuda.is_available():
		device_map = model_config.get("device", "auto")
		# device_map = "auto" if str(device).startswith("cuda") else device
	else:
		device = "cpu"
		device_map = "cpu"

	model, tokenizer = load_llm(
		model_name=model_name,
		cache_dir=model_config["cache_dir"],
		device=device_map,
		dtype=model_config.get("dtype", "float16"),
		trust_remote_code=model_config.get("trust_remote_code", True),
	)
	dataset = load_dataset(
		dataset_name,
		Path(dataset_config["data_dir"]),
	)

	if dataset_name == "fava_annotation":
		results = collect_fava(dataset, model, tokenizer, model_name, device)
	elif dataset_name == "trivia_qa":
		spec = resolve_model_spec(model_name, layer_number)
		results = collect_trivia(
			dataset,
			model,
			tokenizer,
			model_name,
			spec,
			collection_config,
		)
	elif dataset_name == "prover_qa":
		_, response_results = collect_prover_qa(config, model, tokenizer)
		output_path = Path(config.get("proverqa", {}).get(
			"save_dir",
			Path(config["paths"]["result_dir"]) / "proverqa",
		))
		print(f"Saved {len(response_results)} ProverQA responses to {output_path}")
		return output_path
	else:
		raise ValueError(f"Unsupported collector dataset: {dataset_name}")

	output_path = save_results(
		results,
		Path(config["paths"]["result_dir"]),
		model_name,
		dataset_name,
	)
	print(f"Saved collected artifacts to {output_path}")
	return output_path


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="config/experiment.yaml")
	args = parser.parse_args()
	run(load_config(args.config))


if __name__ == "__main__":
	main()

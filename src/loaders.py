import pickle
import json
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset as hf_load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from bs4 import BeautifulSoup

from .classifier import SingleMLP_Classifier
from .config import load_config


def load_hidden_states(path):
    with open(path, "rb") as infile:
        results = pickle.loads(infile.read())
    correct = np.array(results['correct'])
    infile.close()
    return results, correct


def load_llm(model_name, cache_dir, device="auto", dtype="float16", trust_remote_code=True):
    Path(cache_dir).absolute().mkdir(parents=True, exist_ok=True)
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(cache_dir),
                                                 device_map=device, trust_remote_code=trust_remote_code,
                                                 torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir),
                                              trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_classifier_checkpoint(checkpoint_path, input_shape=None, dropout=0.25, device="cpu"):
    """Load a classifier checkpoint and infer its input size when possible."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    weight_key = "linear_relu_stack.0.weight"
    if input_shape is None:
        if weight_key not in checkpoint:
            raise KeyError(f"Missing {weight_key} in checkpoint: {checkpoint_path}")
        input_shape = checkpoint[weight_key].shape[1]
    classifier = SingleMLP_Classifier(input_shape=input_shape, dropout=dropout)
    classifier.load_state_dict(checkpoint)
    return classifier.to(device).eval()


def get_modified_data(data_dir):
    with open(data_dir/"annotations.json", "r", encoding="utf-8") as f:
        data = json.loads(f.read())

    _TAGS = ["entity", "relation", "sentence", "invented", "subjective", "unverifiable"]

    df = {
        "prompt": [],
        "output": [],
        "annotated": [],
        "modified": [],
        "model": [],
        "entity": [],
        "relation": [],
        "sentence": [],
        "invented": [],
        "subjective": [],
        "unverifiable": [],
        "hallucinated": [],
    }

    def modify(s):
        indicator = [0, 0, 0, 0, 0, 0]
        soup = BeautifulSoup(s, "html.parser")
        s1 = ""
        for t in range(len(_TAGS)):
            indicator[t] = len(soup.find_all(_TAGS[t]))
        for elem in soup.find_all(text=True):
            if elem.parent.name != "delete":
                s1 += elem
        return s1, indicator

    for i in range(len(data)):
        df["prompt"].append(data[i]["prompt"])
        df["output"].append(data[i]["output"])
        df["annotated"].append(data[i]["annotated"])
        df["model"].append(data[i]["model"])
        modified_text, indicator = modify(data[i]["annotated"])
        df["modified"].append(modified_text)
        for t in range(len(_TAGS)):
            df[_TAGS[t]].append(indicator[t])
        df["hallucinated"].append(int(sum(indicator) > 0))

    df = pd.DataFrame(df)
    return df


def get_fava_data(n_samples=460, data_dir=None):
    np.random.seed(0)
    data = get_modified_data(data_dir)
    i1 = data["hallucinated"] == 1
    i2 = data["hallucinated"] == 0
    df = pd.concat(
        [
            data[i1][:n_samples],
            data[i2][:n_samples],
        ],
        ignore_index=True,
        sort=False,
    )
    return df


def load_dataset(name, data_dir, start=0, end=10000, fava_samples=460):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    if name == "fava_annotation":
        dataset = get_fava_data(n_samples=fava_samples, data_dir=data_dir)

    elif name == "trivia_qa":
        trivia_qa = hf_load_dataset(
            "mandarjoshi/trivia_qa",
            "rc.nocontext",
            cache_dir=str(data_dir),
        )
        full_dataset = []
        for obs in tqdm(trivia_qa['train'].shuffle(seed=42)):
            aliases = []
            aliases.extend(obs['answer']['aliases'])
            aliases.extend(obs['answer']['normalized_aliases'])
            aliases.append(obs['answer']['value'])
            aliases.append(obs['answer']['normalized_value'])
            full_dataset.append((obs['question'], aliases))
        # only take first 10k over 174k samples
        dataset = full_dataset[start:end]

    elif name == "prover_qa":
        with open(data_dir/"provergen-5000.json", "r") as f:
            dataset = json.load(f)

    else:
        raise ValueError(f"Unknown dataset {name}. Currently, dataset must be one of ['fava_annotation', 'trivia_qa', 'prover_qa']")
    return dataset


def load_components(config):
    """Load the configured LLM, tokenizer, and dataset as one convenience call."""
    model_config = config["model"]
    dataset_config = config["dataset"]
    model, tokenizer = load_llm(
        model_config["name"],
        model_config["cache_dir"],
        device=model_config.get("device", "auto"),
    )
    dataset = load_dataset(
        dataset_config["name"],
        Path(dataset_config["data_dir"]),
        start=dataset_config.get("start", 0),
        end=dataset_config.get("end", 10000),
        fava_samples=dataset_config.get("samples", 460),
    )
    return model, tokenizer, dataset
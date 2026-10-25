import pickle
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_hidden_states(results_file):
    with open(results_file, "rb") as infile:
        results = pickle.loads(infile.read())
    correct = np.array(results['correct'])
    infile.close()
    return results, correct


def model_loader(model_name):
    model_dir = Path("./.cache/models")
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir, trust_remote_code=True)
    return model, tokenizer
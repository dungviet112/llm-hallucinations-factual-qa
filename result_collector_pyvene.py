import functools
from typing import Any, Dict
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict
from functools import partial
from captum.attr import IntegratedGradients
from string import Template
import pyvene as pv


# Data related params
iteration = 0
interval = 400 # We run the inference on these many examples at a time to achieve parallelization
start = iteration * interval
end = start + interval
dataset_name =  "trivia_qa" #"place_of_birth" #"capitals" #"founders"
trex_data_to_question_template = {
    "capitals": Template("What is the capital of $source?"),
    "place_of_birth": Template("Where was $source born?"),
    "founders": Template("Who founded $source?"),
}

# IO
data_dir = Path(".") # Where our data files are stored
model_dir = Path("./.cache/models/") # Cache for huggingface models
results_dir = Path("./results/") # Directory for storing results

# Hardware
gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

# Integrated Grads
ig_steps = 64
internal_batch_size = 4

# Model
model_name = "gpt2" #"llama-2-7b-hf" #"falcon-7b" #"opt-30b"
layer_number = -1


def get_stop_token():
    if "llama" in model_name:
        stop_token = 13
    elif "falcon" in model_name:
        stop_token = 193
    elif "gpt2" in model_name:
        stop_token = 50256
    else:
        stop_token = 50118
    return stop_token


def load_data(dataset_name):
    if dataset_name in trex_data_to_question_template.keys():
        pd_frame = pd.read_csv(data_dir / f'{dataset_name}.csv')
        dataset = [(pd_frame.iloc[i]['subject'], pd_frame.iloc[i]['object'].split("<OR>")) for i in range(start, min(end, len(pd_frame)))]
    elif dataset_name=="trivia_qa":
        trivia_qa = load_dataset('trivia_qa', data_dir='rc.nocontext', cache_dir=str(data_dir))
        full_dataset = []
        for obs in tqdm(trivia_qa['train']):
            aliases = []
            aliases.extend(obs['answer']['aliases'])
            aliases.extend(obs['answer']['normalized_aliases'])
            aliases.append(obs['answer']['value'])
            aliases.append(obs['answer']['normalized_value'])
            full_dataset.append((obs['question'], aliases))
        dataset = full_dataset[start: end]
    else:
        raise ValueError(f"Unknown dataset {dataset_name}.")
    return dataset


def get_next_token(x, model):
    with torch.no_grad():
        return model(x).logits


def generate_response(x, model, *, max_length=100, pbar=False):
    response = []
    bar = tqdm(range(max_length)) if pbar else range(max_length)
    for step in bar:
        logits = get_next_token(x, model)
        next_token = logits.squeeze()[-1].argmax()
        current_ids = torch.concat([x, next_token.view(1, -1)], dim=1)
        response.append(next_token)
        if next_token == get_stop_token() and step > 5:
            break
    return torch.stack(response).cpu().numpy(), logits.squeeze(), current_ids


def answer_question(question, model, tokenizer, *, max_length=100, pbar=False):
    input_ids = tokenizer(question, return_tensors='pt').input_ids.to(model.device)
    response, logits, current_ids = generate_response(input_ids, model, max_length=max_length, pbar=pbar)
    return response, logits, input_ids.shape[-1], current_ids


def answer_trivia(question, targets, model, tokenizer):
    response, logits, start_pos, current_ids = answer_question(question, model, tokenizer)
    str_response = tokenizer.decode(response, skip_special_tokens=True)
    correct = False
    for alias in targets:
        if alias.lower() in str_response.lower():
            correct = True
            break
    return response, str_response, logits, start_pos, correct, current_ids


def answer_trex(source, targets, model, tokenizer, question_template):
    response, logits, start_pos, current_ids = answer_question(question_template.substitute(source=source), model, tokenizer)
    str_response = tokenizer.decode(response, skip_special_tokens=True)
    correct = any([target.lower() in str_response.lower() for target in targets])
    return response, str_response, logits, start_pos, correct, current_ids


def get_start_end_layer(model):
    if "llama" in model_name:
        layer_count = model.model.layers
    elif "falcon" in model_name:
        layer_count = model.transformer.h
    elif "gpt2" in model_name:
        layer_count = model.transformer.h
    else:
        layer_count = model.model.decoder.layers
    layer_st = 0 if layer_number == -1 else layer_number
    layer_en = len(layer_count) if layer_number == -1 else layer_number + 1
    return layer_st, layer_en


def collect_fully_connected(model, current_ids, token_pos, layer_start, layer_end):
    intervention_configs = []
    for layer in range(layer_start, layer_end):
        intervention_configs.append({
            "layer": layer,
            "component": "mlp_output",
            "intervention_type": pv.CollectIntervention
        })
    config = pv.IntervenableConfig(intervention_configs)
    pv_model = pv.IntervenableModel(config, model)

    # collecting at the original boundary of the first generated token
    unit_locations = {"base": token_pos}
    with torch.no_grad():
        _, collected_activations = pv_model(
            base={"input_ids": current_ids},
            unit_locations=unit_locations,
            output_original_output=True,
        )[0]

    # Stack activations from all layers
    first_activation = np.stack([act[0] for act in collected_activations])
    return first_activation


def collect_attention(model, current_ids, token_pos, layer_start, layer_end):
    intervention_configs = []
    for layer in range(layer_start, layer_end):
        intervention_configs.append({
            "layer": layer,
            "component": "attention_output",
            "intervention_type": pv.CollectIntervention
        })
    config = pv.IntervenableConfig(intervention_configs)
    pv_model = pv.IntervenableModel(config, model)
    unit_locations = {"base": token_pos}
    with torch.no_grad():
        _, collected_activations = pv_model(
            base={"input_ids": current_ids},
            unit_locations=unit_locations,
            output_original_output=True,
        )[0]
    first_activation = np.stack([act[0] for act in collected_activations])
    return first_activation


def normalize_attributes(attributes: torch.Tensor) -> torch.Tensor:
    # attributes has shape (batch, sequence size, embedding dim)
    attributes = attributes.squeeze(0)

    # if aggregation == "L2":  # norm calculates a scalar value (L2 Norm)
    norm = torch.norm(attributes, dim=1)
    attributes = norm / torch.sum(norm)  # Normalize the values so they add up to 1
    
    return attributes


def model_forward(input_: torch.Tensor, model, extra_forward_args: Dict[str, Any]) \
            -> torch.Tensor:
    output = model(inputs_embeds=input_, **extra_forward_args)
    return torch.nn.functional.softmax(output.logits[:, -1, :], dim=-1)


def get_embedder(model):
    if "falcon" in model_name:
        return model.transformer.word_embeddings
    elif "opt" in model_name:
        return model.model.decoder.embed_tokens
    elif "llama" in model_name:
        return model.model.embed_tokens
    elif "gpt2" in model_name:
        return model.transformer.wte
    else:
        raise ValueError(f"Unknown model {model_name}")

def get_ig(prompt, forward_func, tokenizer, embedder, model):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    prediction_id = get_next_token(input_ids, model).squeeze()[-1].argmax()
    encoder_input_embeds = embedder(input_ids).detach() # fix this for each model
    ig = IntegratedGradients(forward_func=forward_func)
    attributes = normalize_attributes(
        ig.attribute(
            encoder_input_embeds,
            target=prediction_id,
            n_steps=ig_steps,
            internal_batch_size=internal_batch_size
        )
    ).detach().cpu().numpy()
    return attributes


def compute_and_save_results():
    batch_size = 80
    # Dataset
    dataset = load_data(dataset_name)
    if dataset_name in trex_data_to_question_template.keys():
        question_asker = functools.partial(answer_trex, question_template=trex_data_to_question_template[dataset_name])
    elif dataset_name == "trivia_qa":
        question_asker = answer_trivia
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}.")

    # Model
    model_loader = LlamaForCausalLM if "llama" in model_name else AutoModelForCausalLM
    token_loader = LlamaTokenizer if "llama" in model_name else AutoTokenizer
    tokenizer = token_loader.from_pretrained(f'{model_name}', cache_dir=model_dir)

    model = model_loader.from_pretrained(f'{model_name}',
                                         cache_dir=model_dir,
                                         device_map=device,
                                         torch_dtype=torch.bfloat16,
                                         trust_remote_code=True)
    forward_func = partial(model_forward, model=model, extra_forward_args={})
    embedder = get_embedder(model)

    # Generate results
    results = defaultdict(list)
    for idx in tqdm(range(len(dataset))):
        question, answers = dataset[idx]
        response, str_response, logits, start_pos, correct, current_ids = question_asker(question, answers, model, tokenizer)
        layer_start, layer_end = get_start_end_layer(model)
        first_fully_connected = collect_fully_connected(model, current_ids, start_pos, layer_start, layer_end)
        first_attention = collect_attention(model, current_ids, start_pos, layer_start, layer_end)
        attributes_first = get_ig(question, forward_func, tokenizer, embedder, model)

        results['question'].append(question)
        results['answers'].append(answers)
        results['response'].append(response)
        results['str_response'].append(str_response)
        results['logits'].append(logits.to(torch.float32).cpu().numpy())
        results['start_pos'].append(start_pos)
        results['correct'].append(correct)
        results['first_fully_connected'].append(first_fully_connected)
        results['first_attention'].append(first_attention)
        results['attributes_first'].append(attributes_first)
        if (idx + 1) % batch_size == 0 or idx == len(dataset) - 1:
            with open(results_dir/f"{model_name}_{dataset_name}_batch_{idx//batch_size}.pickle", "wb") as f:
                f.write(pickle.dumps(results))
            results.clear()
    # with open(results_dir/f"{model_name}_{dataset_name}_start-{start}_end-{end}_{datetime.now().month}_{datetime.now().day}.pickle", "wb") as outfile:
    #     outfile.write(pickle.dumps(results))


if __name__ == '__main__':
    compute_and_save_results()
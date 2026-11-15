"""Reusable artifact-generation helpers extracted from result_collector.py."""

from functools import partial
from string import Template

import numpy as np
import torch
from tqdm import tqdm


def get_stop_token(model_name):
    """Return the project stop token for a known model family."""
    name = model_name.lower()
    if "llama" in name:
        return 13
    if "falcon" in name:
        return 193
    if "gpt2" in name:
        return 50256
    if "qwen" in name:
        return 32768
    return 50118


def get_next_token(input_ids, model):
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return outputs.logits, outputs.hidden_states


def generate_response(input_ids, model, model_name, max_length=100, pbar=False):
    response = []
    generated_embeddings = []
    iterator = tqdm(range(max_length), leave=False) if pbar else range(max_length)
    for step in iterator:
        logits, hidden_states = get_next_token(input_ids, model)
        generated_embeddings.append(
            hidden_states[-1][:, -1, :].squeeze(0).float().cpu().numpy()
        )
        next_token = logits.squeeze()[-1].argmax()
        input_ids = torch.cat([input_ids, next_token.view(1, -1)], dim=1)
        response.append(next_token)
        if int(next_token) == get_stop_token(model_name) and step > 5:
            break
    return torch.stack(response).cpu().numpy(), logits.squeeze(), generated_embeddings


def answer_question(question, model, tokenizer, model_name, max_length=100, pbar=False):
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
    response, logits, embeddings = generate_response(
        input_ids, model, model_name, max_length=max_length, pbar=pbar
    )
    return response, tokenizer.decode(response, skip_special_tokens=True), logits, input_ids.shape[-1], embeddings


def answer_trivia(question, targets, model, tokenizer, model_name):
    response, text, logits, start_pos, embeddings = answer_question(
        question, model, tokenizer, model_name
    )
    correct = any(alias.lower() in text.lower() for alias in targets)
    return response, text, logits, start_pos, correct, embeddings


def answer_trex(source, targets, model, tokenizer, model_name, question_template):
    question = question_template.substitute(source=source)
    response, text, logits, start_pos, embeddings = answer_question(
        question, model, tokenizer, model_name
    )
    correct = any(target.lower() in text.lower() for target in targets)
    return response, text, logits, start_pos, correct, embeddings


def normalize_attributes(attributes):
    attributes = attributes.squeeze(0)
    norms = torch.norm(attributes, dim=1)
    return norms / torch.sum(norms)


def model_forward(input_embeddings, model, extra_forward_args=None):
    extra_forward_args = extra_forward_args or {}
    output = model(inputs_embeds=input_embeddings, **extra_forward_args)
    return torch.nn.functional.softmax(output.logits[:, -1, :], dim=-1)


def get_embedder(model, model_name):
    name = model_name.lower()
    if "falcon" in name:
        return model.transformer.word_embeddings
    if "opt" in name:
        return model.model.decoder.embed_tokens
    if "llama" in name or "qwen" in name:
        return model.model.embed_tokens
    if "gpt2" in name:
        return model.transformer.wte
    raise ValueError(f"Unknown model family: {model_name}")


def get_integrated_gradients(prompt, model, tokenizer, model_name, steps=64, batch_size=4):
    """Calculate token attribution for the next-token prediction."""
    from captum.attr import IntegratedGradients

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    logits, _ = get_next_token(input_ids, model)
    prediction_id = logits.squeeze()[-1].argmax()
    embeddings = get_embedder(model, model_name)(input_ids).detach()
    forward = partial(model_forward, model=model, extra_forward_args={})
    attributes = IntegratedGradients(forward).attribute(
        embeddings,
        target=prediction_id,
        n_steps=steps,
        internal_batch_size=batch_size,
    )
    return normalize_attributes(attributes).detach().cpu().numpy()
"""Inference utilities extracted from the application entry point."""

import numpy as np
import torch
from tqdm import tqdm

from .loaders import load_classifier_checkpoint, load_llm


STOP_TOKENS = {13, 128001, 128009}


def load_inference_models(
    model_name,
    cache_dir,
    checkpoint_path,
    device="cpu",
    feature="context_emb",
):
    """Load an LLM/tokenizer and the matching classifier checkpoint."""
    model, tokenizer = load_llm(model_name, cache_dir, device="auto")
    classifier = load_classifier_checkpoint(checkpoint_path, device=device)
    return tokenizer, model, classifier


def generate_response(input_ids, model, tokenizer=None, max_length=50, pbar=False):
    response = []
    generated_embeddings = []
    iterator = tqdm(range(max_length), leave=False) if pbar else range(max_length)
    for step in iterator:
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        last_embedding = outputs.hidden_states[-1][:, -1, :].squeeze(0)
        generated_embeddings.append(last_embedding.float().cpu().numpy())
        next_token = outputs.logits.squeeze()[-1].argmax()
        input_ids = torch.cat([input_ids, next_token.view(1, -1)], dim=1)
        response.append(next_token)
        if int(next_token) in STOP_TOKENS and step > 5:
            break
    response = torch.stack(response).cpu().numpy()
    text = tokenizer.decode(response, skip_special_tokens=True) if tokenizer else ""
    return response, text, outputs.logits.squeeze(), input_ids.shape[-1], generated_embeddings


def infer_hallucination(text, tokenizer, llm_model, classifier_model, feature="context_emb"):
    classifier_model.eval()
    llm_model.eval()
    classifier_device = next(classifier_model.parameters()).device
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(llm_model.device)
    prompt_length = input_ids.shape[-1]
    _, response, _, start_pos, embeddings = generate_response(
        input_ids, llm_model, tokenizer
    )
    if feature == "activation":
        raise NotImplementedError(
            "Activation inference requires a model-specific target module path."
        )
    pooled = torch.tensor(
        np.mean(np.stack(embeddings), axis=0),
        dtype=torch.float32,
        device=classifier_device,
    ).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(classifier_model(pooled), dim=-1)[0]
    label = int(probabilities.argmax())
    return {
        "response": response,
        "correct": label,
        "confidence": float(probabilities[label]),
        "probabilities": {
            "non_hallucinated": float(probabilities[1]),
            "hallucinated": float(probabilities[0]),
        },
    }
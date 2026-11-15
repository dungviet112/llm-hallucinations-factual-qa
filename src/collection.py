"""Model-specific collection helpers shared by artifact collection scripts."""

import re
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from .artifacts import answer_trivia, get_integrated_gradients

# hard-code model specs
MODEL_SPECS = {
    "falcon-40b": ("tiiuae", ".*transformer.h.{layer}.mlp.dense_4h_to_h", ".*transformer.h.{layer}.self_attention.dense", 60),
    "falcon-7b": ("tiiuae", ".*transformer.h.{layer}.mlp.dense_4h_to_h", ".*transformer.h.{layer}.self_attention.dense", 32),
    "open_llama_13b": ("openlm-research", ".*model.layers.{layer}.mlp.up_proj", ".*model.layers.{layer}.self_attn.o_proj", 40),
    "open_llama_7b": ("openlm-research", ".*model.layers.{layer}.mlp.up_proj", ".*model.layers.{layer}.self_attn.o_proj", 32),
    "opt-6.7b": ("facebook", ".*model.decoder.layers.{layer}.fc2", ".*model.decoder.layers.{layer}.self_attn.out_proj", 32),
    "opt-30b": ("facebook", ".*model.decoder.layers.{layer}.fc2", ".*model.decoder.layers.{layer}.self_attn.out_proj", 48),
    "llama-2-7b-hf": ("meta-llama", ".*model.layers.{layer}.mlp.up_proj", ".*model.layers.{layer}.self_attn.o_proj", 32),
    "gpt2": ("openai-community", ".*transformer.h.{layer}.mlp.c_proj", ".*transformer.h.{layer}.attn.c_proj", 12),
    "Qwen2.5-7B-Instruct": ("Qwen", ".*model.layers.{layer}.mlp.up_proj", ".*model.layers.{layer}.self_attn.o_proj", 28),
    "Qwen3-8B": ("Qwen", ".*model.layers.{layer}.mlp.up_proj", ".*model.layers.{layer}.self_attn.o_proj", 36),
    "Meta-Llama-3-8B-Instruct": ("meta-llama", ".*model.layers.{layer}.mlp.up_proj", ".*model.layers.{layer}.self_attn.o_proj", 32),
    "Llama-3.1-8B-Instruct": ("meta-llama", ".*model.layers.{layer}.mlp.up_proj", ".*model.layers.{layer}.self_attn.o_proj", 32),
    "DeepSeek-R1-Distill-Llama-8B": ("deepseek-ai", ".*model.layers.{layer}.mlp.up_proj", ".*model.layers.{layer}.self_attn.o_proj", 32),
}


def model_key(model_name):
    short_name = model_name.rstrip("/").split("/")[-1]
    if short_name in MODEL_SPECS:
        return short_name
    raise ValueError(f"Unsupported model family: {model_name}")


def resolve_model_spec(model_name, layer_number=-1):
    key = model_key(model_name)
    owner, fc_pattern, attention_pattern, layer_count = MODEL_SPECS[key]
    if not -layer_count <= layer_number < layer_count:
        raise ValueError(f"Layer {layer_number} is invalid for {model_name} ({layer_count} layers).")
    layer_pattern = "[0-9]+" if layer_number == -1 else str(layer_number)
    return {
        "key": key,
        "repo": f"{owner}/{key}",
        "fc_pattern": fc_pattern.format(layer=layer_pattern),
        "attention_pattern": attention_pattern.format(layer=layer_pattern),
        "layer_start": 0 if layer_number == -1 else layer_number,
        "layer_end": layer_count if layer_number == -1 else layer_number + 1,
    }


class ActivationStore:
    """Collect hook outputs and retrieve the first generated-token activations."""

    def __init__(self):
        self.fully_connected = defaultdict(list)
        self.attention = defaultdict(list)
        self.handles = []

    def clear(self):
        self.fully_connected.clear()
        self.attention.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def register(self, model, spec):
        def save_fc(name, _module, _inputs, output):
            self.fully_connected[name].append(
                output.squeeze().detach().float().cpu().numpy()
            )

        def save_attention(name, _module, _inputs, output):
            self.attention[name].append(
                output.squeeze().detach().float().cpu().numpy()
            )

        for name, module in model.named_modules():
            if re.match(f"{spec['fc_pattern']}$", name):
                self.handles.append(module.register_forward_hook(
                    lambda module, inputs, output, name=name: save_fc(name, module, inputs, output)
                ))
            if re.match(f"{spec['attention_pattern']}$", name):
                self.handles.append(module.register_forward_hook(
                    lambda module, inputs, output, name=name: save_attention(name, module, inputs, output)
                ))

    def _values_for_layers(self, values, pattern, token_pos, layer_start, layer_end):
        match = re.match(r"(.*?)(?:\[0-9\]+|\d+)(.*)", pattern)
        if match is None:
            raise ValueError(f"Cannot determine layer pattern: {pattern}")
        prefix, suffix = match.groups()
        if prefix.startswith(".*"):
            prefix = prefix[2:]
        return np.stack([
            values[f"{prefix}{layer}{suffix}"][-1][token_pos]
            for layer in range(layer_start, layer_end)
        ])

    def collect(self, token_pos, spec):
        return (
            self._values_for_layers(
                self.fully_connected,
                spec["fc_pattern"],
                token_pos,
                spec["layer_start"],
                spec["layer_end"],
            ),
            self._values_for_layers(
                self.attention,
                spec["attention_pattern"],
                token_pos,
                spec["layer_start"],
                spec["layer_end"],
            ),
        )


def answer_fava(model_name, data, model, tokenizer, device=None):
    """Collect contextual embeddings and labels for FAVA annotations."""
    from fastchat.model import get_conversation_template

    device = device or model.device
    embeddings, labels = [], []
    for row_index in tqdm(range(len(data))):
        row = data.iloc[row_index]
        chat_template = get_conversation_template(model_name)
        chat_template.set_system_message("")
        chat_template.messages = []
        chat_template.append_message(chat_template.roles[0], row["prompt"].strip())
        chat_template.append_message(chat_template.roles[1], row["output"].strip())
        full_prompt = chat_template.get_prompt()
        user_prompt = full_prompt.split(row["output"].strip())[0].strip()
        prompt_len = tokenizer(
            user_prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids.shape[1]
        tokenized = tokenizer(
            full_prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids
        with torch.no_grad():
            outputs = model(
                tokenized.to(device),
                output_hidden_states=True,
                return_dict=True,
                output_attentions=True,
                use_cache=False,
            )
        embeddings.append(
            outputs.hidden_states[-1][0, prompt_len:, :].float().cpu()
        )
        labels.append(int(row["hallucinated"] == 1))
    return embeddings, labels


__all__ = [
    "ActivationStore",
    "MODEL_SPECS",
    "answer_fava",
    "answer_trivia",
    "get_integrated_gradients",
    "model_key",
    "resolve_model_spec",
]
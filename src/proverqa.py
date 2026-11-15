"""ProverQA generation and embedding collection helpers."""

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


DEFAULT_SYSTEM_PROMPT = "Given a problem statement as contexts, the task is to answer a logical reasoning question. Your answer should be in JSON format with key: answer."

DEFAULT_PROMPT_TEMPLATE = """Context:
Anyone who can solve problems and communicate effectively has practical skills.
Paola appreciates beauty. If Paola appreciates beauty, then she enjoys manual work.
Paola is either skilled in mathematics or has practical skills, but not both.
Paola has practical skills.

Question: Based on the above information, is the following statement true, false, or uncertain? Paola is skilled in mathematics.

Options:
A) True
B) False
C) Uncertain

The correct option is: {
  \"answer\": \"B\"
}
------
[[INSTRUCTION]]

The correct option is:"""

DEFAULT_REASONING_SYSTEM_PROMPT = "Given a problem statement as contexts, the task is to answer a logical reasoning question. Your answer should be in JSON format with keys: reasoning, answer."

DEFAULT_REASONING_PROMPT_TEMPLATE = """Context:
Anyone who can solve problems and communicate effectively has practical skills.
Paola appreciates beauty. If Paola appreciates beauty, then she enjoys manual work.
Paola has practical skills.

Question: Based on the above information, is the following statement true, false, or uncertain? Paola is artistically inclined.

Options:
A) True
B) False
C) Uncertain

The correct option is: {
  \"reasoning\": \"The context does not establish that Paola is artistically inclined.\",
  \"answer\": \"C\"
}
------
[[INSTRUCTION]]

The correct option is:"""


def get_prompt_config(reasoning=False, system_prompt=None, prompt_template=None):
    """Return ProverQA prompts, allowing YAML values to override defaults."""
    if reasoning:
        default_system = DEFAULT_REASONING_SYSTEM_PROMPT
        default_template = DEFAULT_REASONING_PROMPT_TEMPLATE
    else:
        default_system = DEFAULT_SYSTEM_PROMPT
        default_template = DEFAULT_PROMPT_TEMPLATE
    return (
        system_prompt or default_system,
        prompt_template or default_template,
    )


def extract_answer(trace):
    match = re.search(r'"answer"\s*:\s*"([^"]+)"', trace)
    return match.group(1) if match else None


def collect_proverqa(
    model,
    tokenizer,
    json_path,
    save_dir,
    system_prompt,
    prompt_template,
    reasoning=False,
    max_new_tokens=128,
):
    """Generate ProverQA answers and save responses plus hidden embeddings."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with Path(json_path).open(encoding="utf-8") as file:
        data = json.load(file)
    results = defaultdict(list)
    response_results = []
    for index, item in enumerate(tqdm(data)):
        gold_label = item["output"][-4]
        instruction = item["instruction"]
        full_prompt = prompt_template.replace("[[INSTRUCTION]]", instruction)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ]
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=reasoning,
        ).to(model.device)
        outputs = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_length = tokenized.input_ids.shape[1]
        trace = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        extracted_answer = extract_answer(trace)
        correct = extracted_answer == gold_label
        response_results.append({
            "instruction": instruction,
            "model_trace": trace,
            "pred_label": extracted_answer,
            "gold_label": gold_label,
            "correct_prediction": correct,
        })
        if extracted_answer is None:
            continue
        full_messages = messages + [{"role": "assistant", "content": trace}]
        full_inputs = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)
        with torch.no_grad():
            hidden_states = model(
                input_ids=full_inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[-1]
        embeddings = hidden_states[0, prompt_length - 1:-1].float().cpu().numpy()
        results["generated_embeddings"].append(np.stack(embeddings))
        results["index"].append(index)
        results["correct"].append(correct)
    with (save_dir / "logical_result_standard.json").open("w", encoding="utf-8") as file:
        json.dump(response_results, file, indent=2)
    with (save_dir / "contextual_embs_standard_all_gen_tokens.pkl").open("wb") as file:
        pickle.dump(results, file)
    return results, response_results
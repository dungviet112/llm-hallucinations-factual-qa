import json
import os
import re
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

data_dir = Path("./data")
model_dir = Path("./.cache/models")
save_dir = Path("./results/5000_samples_ProverQA_Llama3.1-8B")

reasoning = True
do_sample = False
temperature = 0.0
if temperature != 0.0:
    do_sample = True

if reasoning:
    max_new_tokens = 1024
    system_prompt = "Given a problem statement as contexts, the task is to answer a logical reasoning question. Your answer should be in JSON format with keys: reasoning, answer."
    in_context_examples_for_train = "Context:\nAnyone who can solve problems and communicate effectively has practical skills. If Paola appreciates beauty, then she values precision and enjoys manual work. Paola appreciates beauty. Paola is either skilled in mathematics or has practical skills, but not both. If someone enjoys manual work and values attention to detail, then they can craft with their hands. Paola is either skilled in mathematics or artistically inclined, but surprisingly, she is actually both. Paola values attention to detail. Anyone who crafts with their hands has practical skills.\n\nQuestion: Based on the above information, is the following statement true, false, or uncertain? Paola is artistically inclined.\n\nOptions:\nA) True\nB) False\nC) Uncertain\n\nThe correct option is: {\n  \"reasoning\": \"Paola appreciates beauty. If Paola appreciates beauty, then she values precision and enjoys manual work. Paola enjoys manual work. Paola enjoys manual work. Paola values attention to detail. If someone enjoys manual work and values attention to detail, then they can craft with their hands. Paola crafts with her hands. Paola crafts with her hands. Anyone who crafts with their hands has practical skills. Paola has practical skills. Paola has practical skills. Paola is either skilled in mathematics or has practical skills, but not both. Paola is not skilled in mathematics. Paola is not skilled in mathematics. Paola is either skilled in mathematics or artistically inclined, but surprisingly, she is actually both. Paola is artistically inclined. Therefore, it is true that Paola is artistically inclined. The correct option is: A.\",\n  \"answer\": \"A\"\n}\n------\nContext:\nIf Priscilla loves nature or shares beautiful images (or both), then she inspires environmental awareness. Priscilla either portrays cityscapes or captures sunsets, but not both. Priscilla documents wildlife. If Priscilla has an urban focus, then she will portray cityscapes. If Priscilla travels extensively, then she either takes landscape photos or captures sunsets (or both). Priscilla has an urban focus. Everyone who either documents wildlife or captures sunsets (but not both) shares beautiful images.\n\nQuestion: Based on the above information, is the following statement true, false, or uncertain? Priscilla does not inspire environmental awareness.\n\nOptions:\nA) True\nB) False\nC) Uncertain\n\nThe correct option is: {\n  \"reasoning\": \"Priscilla has an urban focus. If Priscilla has an urban focus, then she will portray cityscapes. Priscilla portrays cityscapes. Priscilla portrays cityscapes. Priscilla either portrays cityscapes or captures sunsets, but not both. Priscilla does not capture sunsets. Priscilla documents wildlife. Priscilla does not capture sunsets. Everyone who either documents wildlife or captures sunsets (but not both) shares beautiful images. Priscilla shares beautiful images. Priscilla shares beautiful images. If Priscilla loves nature or shares beautiful images (or both), then she inspires environmental awareness. Priscilla inspires environmental awareness. Therefore, it is false that Priscilla does not inspire environmental awareness. The correct option is: B.\",\n  \"answer\": \"B\"\n}\n------\n[[INSTRUCTION]]\n\nThe correct option is:"
else:
    max_new_tokens = 128
    system_prompt = "Given a problem statement as contexts, the task is to answer a logical reasoning question. Your answer should be in JSON format with key: answer."
    in_context_examples_for_train = "Context:\nAnyone who can solve problems and communicate effectively has practical skills. If Paola appreciates beauty, then she values precision and enjoys manual work. Paola appreciates beauty. Paola is either skilled in mathematics or has practical skills, but not both. If someone enjoys manual work and values attention to detail, then they can craft with their hands. Paola is either skilled in mathematics or artistically inclined, but surprisingly, she is actually both. Paola values attention to detail. Anyone who crafts with their hands has practical skills.\n\nQuestion: Based on the above information, is the following statement true, false, or uncertain? Paola is artistically inclined.\n\nOptions:\nA) True\nB) False\nC) Uncertain\n\nThe correct option is: {\n  \"answer\": \"A\"\n}\n------\nContext:\nIf Priscilla loves nature or shares beautiful images (or both), then she inspires environmental awareness. Priscilla either portrays cityscapes or captures sunsets, but not both. Priscilla documents wildlife. If Priscilla has an urban focus, then she will portray cityscapes. If Priscilla travels extensively, then she either takes landscape photos or captures sunsets (or both). Priscilla has an urban focus. Everyone who either documents wildlife or captures sunsets (but not both) shares beautiful images.\n\nBased on the above information, is the following statement true, false, or uncertain? Priscilla does not inspire environmental awareness.\n\nOptions:\nA) True\nB) False\nC) Uncertain\n\nThe correct option is: {\n  \"answer\": \"B\"\n}\n------\n[[INSTRUCTION]]\n\nThe correct option is:"

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir, trust_remote_code=True)

json_path = data_dir / "ProverQA/train/provergen-5000.json"
with open(json_path, "r") as f:
    data = json.load(f)

# in_context_examples_for_test = "Context:\nAnyone who can solve problems and communicate effectively has practical skills. If Paola appreciates beauty, then she values precision and enjoys manual work. Paola appreciates beauty. Paola is either skilled in mathematics or has practical skills, but not both. If someone enjoys manual work and values attention to detail, then they can craft with their hands. Paola is either skilled in mathematics or artistically inclined, but surprisingly, she is actually both. Paola values attention to detail. Anyone who crafts with their hands has practical skills.\n\nQuestion: Based on the above information, is the following statement true, false, or uncertain? Paola is artistically inclined.\n\nOptions:\nA) True\nB) False\nC) Uncertain\n\nThe correct option is: {\n  \"answer\": \"A\"\n}\n------\nContext:\nIf Priscilla loves nature or shares beautiful images (or both), then she inspires environmental awareness. Priscilla either portrays cityscapes or captures sunsets, but not both. Priscilla documents wildlife. If Priscilla has an urban focus, then she will portray cityscapes. If Priscilla travels extensively, then she either takes landscape photos or captures sunsets (or both). Priscilla has an urban focus. Everyone who either documents wildlife or captures sunsets (but not both) shares beautiful images.\n\nBased on the above information, is the following statement true, false, or uncertain? Priscilla does not inspire environmental awareness.\n\nOptions:\nA) True\nB) False\nC) Uncertain\n\nThe correct option is: {\n  \"answer\": \"B\"\n}\n------\nContext:\n[[CONTEXT]]\n\nQuestion: [[QUESTION]]\n\nOptions:\n[[OPTIONS]]\n\nThe correct option is:"

all_results = []

for item in tqdm(data):
    gold_label = item['output'][-4]
    instruction = item['instruction']
    
    full_prompt = in_context_examples_for_train.replace("[[INSTRUCTION]]", instruction)
    
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': full_prompt}
    ]

    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False
    ).to(model.device)

    outputs = model.generate(
        **tokenized_chat,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )

    prompt_length = len(tokenized_chat.input_ids[0])
    trace = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

    pattern = r'"answer"\s*:\s*"([^"]+)"'
    match = re.search(pattern, trace)
    if match:
        extracted_answer = match.group(1)
    else:
        extracted_answer = None

    correct = extracted_answer == gold_label

    # print(f"Trace: {trace}")
    # print(f"Gold label: {gold_label}")
    # print(f"Extracted answer: {extracted_answer}")
    # print(f"Correct: {correct}")
    # print("*" * 50)

    eval_result = {
        "instruction": instruction,
        "model_trace": trace,
        "pred_label": extracted_answer,
        "gold_label": gold_label,
        "correct_prediction": correct,
    }
    all_results.append(eval_result)
    
save_dir.mkdir(parents=True, exist_ok=True)
with open(save_dir / f"logical_result_reasoning.json", "w") as f:
    json.dump(all_results, f, indent=2)
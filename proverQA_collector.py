import json
import pickle
import os
import re
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
from collections import defaultdict
import argparse
from utils import model_loader


def proverqa_answer(model, tokenizer, json_path, save_dir, reasoning, max_new_tokens):
    with open(json_path, "r") as f:
        data = json.load(f)

    results = defaultdict(list)
    response_results = []
    index = 0

    for item in tqdm(data):
        gold_label = item['output'][-4]
        instruction = item['instruction']
        
        full_prompt = in_context_examples_for_train.replace("[[INSTRUCTION]]", instruction)
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': full_prompt}
        ]

        tokenized_query = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=reasoning
        ).to(model.device)
        print(f"Tokenized query input_ids shape: {tokenized_query.input_ids.shape}")

        outputs = model.generate(
            **tokenized_query,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"Generated outputs shape: {outputs.shape}")

        prompt_length = len(tokenized_query.input_ids[0])
        trace = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

        pattern = r'"answer"\s*:\s*"([^"]+)"'
        match = re.search(pattern, trace)
        if match:
            extracted_answer = match.group(1)
        else:
            extracted_answer = None

        correct = extracted_answer == gold_label

        eval_result = {
            "instruction": instruction,
            "model_trace": trace,
            "pred_label": extracted_answer,
            "gold_label": gold_label,
            "correct_prediction": correct,
        }
        response_results.append(eval_result)

        if extracted_answer is not None:
            full_messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': full_prompt},
                {'role': 'assistant', 'content': trace}
            ]

            full_inputs = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            ).input_ids.to(model.device)

            kwargs = {
                "input_ids": full_inputs,
                "use_cache": False,
                "past_key_values": None,
                "output_hidden_states": True,
                "return_dict": True,
            }
            with torch.no_grad():
                outputs = model(**kwargs)

            generated_embeddings = outputs.hidden_states[-1][0][prompt_length-1:-1].to(torch.float32).detach().cpu().numpy()
            generated_embeddings = np.stack(generated_embeddings)

            results['generated_embeddings'].append(generated_embeddings)
            results['index'].append(index)
            results['correct'].append(correct)
        index += 1
    # save repsonse results
    with open(save_dir / "logical_result_standard.json", "w") as f:
        json.dump(response_results, f, indent=2)

    # save embeddings
    with open(save_dir / "contextual_embs_standard_all_gen_tokens.pkl", "wb") as outfile:
        outfile.write(pickle.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Name of the model to use.")
    parser.add_argument("--json_path", type=str, default="./data/ProverQA/train/provergen-5000.json", help="Path to the JSON data file.")
    parser.add_argument("--save_dir", type=str, default="./results/5000_samples_ProverQA_Llama3.1-8B", help="Path to save the embeddings.")
    parser.add_argument("--reasoning", action="store_true", help="Whether to include reasoning in the output.")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.reasoning:
        max_new_tokens = 1024
        system_prompt = "Given a problem statement as contexts, the task is to answer a logical reasoning question. Your answer should be in JSON format with keys: reasoning, answer."
        in_context_examples_for_train = "Context:\nAnyone who can solve problems and communicate effectively has practical skills. If Paola appreciates beauty, then she values precision and enjoys manual work. Paola appreciates beauty. Paola is either skilled in mathematics or has practical skills, but not both. If someone enjoys manual work and values attention to detail, then they can craft with their hands. Paola is either skilled in mathematics or artistically inclined, but surprisingly, she is actually both. Paola values attention to detail. Anyone who crafts with their hands has practical skills.\n\nQuestion: Based on the above information, is the following statement true, false, or uncertain? Paola is artistically inclined.\n\nOptions:\nA) True\nB) False\nC) Uncertain\n\nThe correct option is: {\n  \"reasoning\": \"Paola appreciates beauty. If Paola appreciates beauty, then she values precision and enjoys manual work. Paola enjoys manual work. Paola enjoys manual work. Paola values attention to detail. If someone enjoys manual work and values attention to detail, then they can craft with their hands. Paola crafts with her hands. Paola crafts with her hands. Anyone who crafts with their hands has practical skills. Paola has practical skills. Paola has practical skills. Paola is either skilled in mathematics or has practical skills, but not both. Paola is not skilled in mathematics. Paola is not skilled in mathematics. Paola is either skilled in mathematics or artistically inclined, but surprisingly, she is actually both. Paola is artistically inclined. Therefore, it is true that Paola is artistically inclined. The correct option is: A.\",\n  \"answer\": \"A\"\n}\n------\nContext:\nIf Priscilla loves nature or shares beautiful images (or both), then she inspires environmental awareness. Priscilla either portrays cityscapes or captures sunsets, but not both. Priscilla documents wildlife. If Priscilla has an urban focus, then she will portray cityscapes. If Priscilla travels extensively, then she either takes landscape photos or captures sunsets (or both). Priscilla has an urban focus. Everyone who either documents wildlife or captures sunsets (but not both) shares beautiful images.\n\nQuestion: Based on the above information, is the following statement true, false, or uncertain? Priscilla does not inspire environmental awareness.\n\nOptions:\nA) True\nB) False\nC) Uncertain\n\nThe correct option is: {\n  \"reasoning\": \"Priscilla has an urban focus. If Priscilla has an urban focus, then she will portray cityscapes. Priscilla portrays cityscapes. Priscilla portrays cityscapes. Priscilla either portrays cityscapes or captures sunsets, but not both. Priscilla does not capture sunsets. Priscilla documents wildlife. Priscilla does not capture sunsets. Everyone who either documents wildlife or captures sunsets (but not both) shares beautiful images. Priscilla shares beautiful images. Priscilla shares beautiful images. If Priscilla loves nature or shares beautiful images (or both), then she inspires environmental awareness. Priscilla inspires environmental awareness. Therefore, it is false that Priscilla does not inspire environmental awareness. The correct option is: B.\",\n  \"answer\": \"B\"\n}\n------\n[[INSTRUCTION]]\n\nThe correct option is:"
    else:
        max_new_tokens = 128
        system_prompt = "Given a problem statement as contexts, the task is to answer a logical reasoning question. Your answer should be in JSON format with key: answer."
        in_context_examples_for_train = "Context:\nAnyone who can solve problems and communicate effectively has practical skills. If Paola appreciates beauty, then she values precision and enjoys manual work. Paola appreciates beauty. Paola is either skilled in mathematics or has practical skills, but not both. If someone enjoys manual work and values attention to detail, then they can craft with their hands. Paola is either skilled in mathematics or artistically inclined, but surprisingly, she is actually both. Paola values attention to detail. Anyone who crafts with their hands has practical skills.\n\nQuestion: Based on the above information, is the following statement true, false, or uncertain? Paola is artistically inclined.\n\nOptions:\nA) True\nB) False\nC) Uncertain\n\nThe correct option is: {\n  \"answer\": \"A\"\n}\n------\nContext:\nIf Priscilla loves nature or shares beautiful images (or both), then she inspires environmental awareness. Priscilla either portrays cityscapes or captures sunsets, but not both. Priscilla documents wildlife. If Priscilla has an urban focus, then she will portray cityscapes. If Priscilla travels extensively, then she either takes landscape photos or captures sunsets (or both). Priscilla has an urban focus. Everyone who either documents wildlife or captures sunsets (but not both) shares beautiful images.\n\nBased on the above information, is the following statement true, false, or uncertain? Priscilla does not inspire environmental awareness.\n\nOptions:\nA) True\nB) False\nC) Uncertain\n\nThe correct option is: {\n  \"answer\": \"B\"\n}\n------\n[[INSTRUCTION]]\n\nThe correct option is:"

    try:
        model, tokenizer = model_loader(args.model)
        proverqa_answer(model, tokenizer, args.json_path, save_dir, args.reasoning, max_new_tokens)
    except Exception as e:
        print(e)
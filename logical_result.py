import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForMultimodalLM
from pathlib import Path
import torch
import datasets
from tqdm import tqdm
print("Is CUDA available?", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

max_new_tokens = 128
do_sample = False
temperature = 0.0
if do_sample:
    temperature = 0.3

model_dir = Path("./.cache/models")
data_dir = Path("./data")
save_dir = Path("./results/2100_samples_SATBench_Llama3.1-8B")

data = datasets.load_dataset("LLM4Code/SATBench", split="train", cache_dir=data_dir)

model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "Qwen/Qwen3-8B"
if "gemma" in model_name:
    model = AutoModelForMultimodalLM.from_pretrained(model_name, cache_dir=model_dir,
                                                    device_map="auto", dtype="auto",
                                                    trust_remote_code=True)
    tokenizer = AutoProcessor.from_pretrained(model_name)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir, trust_remote_code=True)

all_results = []

for i in tqdm(range(len(data["scenario"]))):
    scenario = data['scenario'][i]
    conditions = data['conditions'][i]
    question = data['question'][i]
#     system_prompt = f"""You are a logical reasoning assistant. You are given a logic puzzle.
# Guidelines:
# - All constraints come **only** from the <conditions> section.
# - The <scenario> provides background and intuition, but **does not impose any additional rules or constraints**.
# - All variables represent **independent decisions**; there is no mutual exclusivity or implicit linkage unless stated explicitly in <conditions>.
# - Variables not mentioned in <conditions> are considered unknown and irrelevant to satisfiability.
# Your task:
# - If the puzzle is satisfiable, propose one valid assignment that satisfies all the conditions.
# - If the puzzle is unsatisfiable, explain why some of the conditions cannot all be true at once.
# Think step by step but don't return any intermediate reasoning, summarize the key points, and only give a brief response.
# Do not use conversational filler, introductions, or transitional phrasing like 'Let's look at this' or 'Let's analyze the conditions'. Start immediately with the core facts.
# At the end of your response, output exactly one of the following labels on a new line:
# [SAT] - if a valid assignment exists
# [UNSAT] - if the constraints cannot be satisfied
# Do not add any text or formatting after the final label.
# """
    system_prompt = f"""You are a logical reasoning assistant. You are given a logic puzzle.
Guidelines:
- All constraints come **only** from the <conditions> section.
- The <scenario> provides background and intuition, but **does not impose any additional rules or constraints**.
- All variables represent **independent decisions**; there is no mutual exclusivity or implicit linkage unless stated explicitly in <conditions>.
- Variables not mentioned in <conditions> are considered unknown and irrelevant to satisfiability.
Your task: Think step by step but don't return any intermediate reasoning, and only give a brief response following these rules:
- If the puzzle is satisfiable, response only the final label [SAT].
- If the puzzle is unsatisfiable, response only the final label [UNSAT].
Do not add any text or formatting after the final label.
"""
    user_prompt = f"""<scenario>
{scenario}
<conditions>
{conditions}
<question>
{question}"""

    messages = [
        {
            "role": "system", 
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": user_prompt
        }
    ]

    # 4. Apply the chat template to automatically inject the correct model tags
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Tells the model it is its turn to reply
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False
    ).to(model.device)

    # 5. Generate the response
    outputs = model.generate(
        **tokenized_chat,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    # 6. Decode and print the result
    # We slice outputs to only decode tokens generated *after* your prompt
    # prompt_length = tokenized_chat.shape[1]
    prompt_length = len(tokenized_chat.input_ids[0])
    trace = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    # input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(model.device)
    # with torch.no_grad():
    #     outputs = model.generate(input_ids, temperature=temperature, 
    #                             # top_p=0.9, top_k=20,
    #                             pad_token_id=tokenizer.eos_token_id,
    #                             early_stopping=True,
    #                             # repetition_penalty=1.1,
    #                             do_sample=do_sample,
    #                             max_new_tokens=max_new_tokens)
    #     trace = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    trace = trace.strip()
    # print(f"Trace: {trace}")
    text = trace.upper()
    tokens = re.findall(r"\b[A-Z]{3,6}\b", text)  # e.g., "SAT", "UNSAT"
    # match = re.search(r"<verdict>\s*(SAT|UNSAT)\s*</verdict>\s*$", text, re.IGNORECASE)
    # matches = re.findall(r"<verdict>\s*(SAT|UNSAT)\s*</verdict>", trace, re.IGNORECASE)
    # print(matches)
    # if matches:
    #     pred_label = matches[-1].upper()
    # else:
    #     pred_label = None
    if "UNSAT" in tokens:
        pred_label = "UNSAT"
    elif "SAT" in tokens:
        pred_label = "SAT"
    else:
        pred_label = None
    # print(f"Predicted label: {pred_label}")

    gold_label = "SAT" if data["satisfiable"][i] else "UNSAT"
    correct = pred_label == gold_label
    # print(f"Gold label: {gold_label}")
    # print(f"Correct: {correct}")
    # print("*" * 50)

    eval_result = {
        "scenario": scenario,
        "conditions": conditions,
        "question": question,
        "recovered_formula": data["recovered_formula"][i],
        "variables": data["variable_mapping"][i],
        "model_trace": trace,
        "pred_label": pred_label,
        "gold_label": gold_label,
        "correct_prediction": correct,
    }
    all_results.append(eval_result)

save_dir.mkdir(parents=True, exist_ok=True)
with open(save_dir / f"logical_result_max{max_new_tokens}_temp{temperature}_verdict_only.json", "w") as f:
    json.dump(all_results, f, indent=2)

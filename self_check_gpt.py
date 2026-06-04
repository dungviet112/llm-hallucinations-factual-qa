from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
import functools
import pickle
import logging
from datetime import datetime
import os

from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckBERTScore, SelfCheckNgram
from sklearn.metrics import roc_auc_score
import statistics
import spacy

from result_collector import trex_data_to_question_template, answer_trivia, answer_trex, load_data, model_dir

import torch
import numpy as np

# Disable transformers progress bar
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

log_filename = f"./logs/selfcheck_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Save & show WARNING and ERROR only
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.WARNING)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

org="openlm-research"
model_name = "open_llama_7b"
repo = f"{org}/{model_name}"

# Data related params
dataset_name = "trivia_qa"

# GPU
gpu = "1"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

# SelfCheckGPT
self_checkgpt_temperature = 0.1
selfcheckgpt_n_trials = 10

dataset = load_data(dataset_name)

tokenizer = LlamaTokenizer.from_pretrained(repo)
model = LlamaForCausalLM.from_pretrained(repo, cache_dir=model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
selfcheck_ngram = SelfCheckNgram(n=1) # n=1 means Unigram

def generate_responses(question, str_response, tokenizer):
    # generate several responses to the question and (self)check them against the zero temp response
    inputs = tokenizer(question, return_tensors="pt").input_ids.to(device)
    start_pos = inputs.size(dim=-1)

    hitemp_str_responses = []
    for i in range(0, selfcheckgpt_n_trials):
        model_outputs = model.generate(
            inputs, do_sample=True, temperature=self_checkgpt_temperature, max_new_tokens=100, return_dict_in_generate=True, output_scores=True
        )
        generated_tokens_ids = model_outputs.sequences[0]
        generated_text = tokenizer.decode(generated_tokens_ids[start_pos:]).replace("\n", " ").strip()
        if generated_text:
            hitemp_str_responses.append(generated_text)

    # Handle case where no valid responses were generated
    if not hitemp_str_responses:
        logger.warning(f"No valid responses generated for question: {question}")
        return [], [], [], []

    selfcheck_scores_bert_overall = []
    selfcheck_scores_bert_average = []
    selfcheck_ngram_overall = []
    
    sentences = [str_response]
    overall_bertscore = selfcheck_bertscore.predict(
        sentences = sentences,                   # list of sentences
        sampled_passages = hitemp_str_responses, # list of sampled passages
    )
    selfcheck_scores_bert_overall.append(overall_bertscore[0])
    
    nlp = spacy.load("en_core_web_sm")
    sentences = [sent for sent in nlp(str_response).sents]
    sentences = [sent.text.strip() for sent in sentences if len(sent) > 3]
    
    # Handle case where response has no valid sentences
    if sentences:
        all_bertscores = selfcheck_bertscore.predict(
            sentences = sentences,                          # list of sentences
            sampled_passages = hitemp_str_responses, # list of sampled passages
        )
        average_bertscore = statistics.mean(all_bertscores)
        selfcheck_scores_bert_average.append(average_bertscore)
    else:
        selfcheck_scores_bert_average.append(overall_bertscore[0])
      
    
    sent_scores_ngram = selfcheck_ngram.predict(
        sentences = sentences if sentences else [str_response],   
        passage = str_response,
        sampled_passages = hitemp_str_responses,
    )
    selfcheck_ngram_overall.append(sent_scores_ngram)
    
    return hitemp_str_responses, selfcheck_scores_bert_overall, selfcheck_scores_bert_average, selfcheck_ngram_overall

selfcheck_dict = {
        'question': [],
        'response': [],
        'str_response': [],
        'start_pos': [],
        'correct': [],
        'hitemp_str_responses': [],
        'selfcheck_scores_bert_overall': [],
        'selfcheck_scores_bert_average': [],
        'selfcheck_ngram_overall': []
    }

selfcheck_arr_overall = []
selfcheck_arr_average = []
selfcheck_ngram_average = []
correct_arr = []

if dataset_name in trex_data_to_question_template.keys():
    question_asker = functools.partial(answer_trex, question_template=trex_data_to_question_template[dataset_name])
elif dataset_name == "trivia_qa":
    question_asker = answer_trivia
else:
    raise ValueError(f"Unknown dataset name {dataset_name}.")

for idx in tqdm(range(len(dataset))):
    try:
        question, answers = dataset[idx]
        response, str_response, logits, start_pos, correct, _ = question_asker(question, answers, model, tokenizer)
        hitemp_str_responses, selfcheck_scores_bert_overall, selfcheck_scores_bert_average, selfcheck_ngram_overall\
            = generate_responses(
                question if dataset_name=="trivia_qa" else trex_data_to_question_template[dataset_name].substitute(source=question),
                str_response, 
                tokenizer
            )
        
        if not hitemp_str_responses:
            continue

        selfcheck_dict['question'].append(question)
        selfcheck_dict['response'].append(response)
        selfcheck_dict['str_response'].append(str_response)
        selfcheck_dict['start_pos'].append(start_pos)
        selfcheck_dict['correct'].append(correct)
        selfcheck_dict['hitemp_str_responses'].append(hitemp_str_responses)
        selfcheck_dict['selfcheck_scores_bert_overall'].append(selfcheck_scores_bert_overall)
        selfcheck_dict['selfcheck_scores_bert_average'].append(selfcheck_scores_bert_average)
        selfcheck_dict['selfcheck_ngram_overall'].append(selfcheck_ngram_overall)

        selfcheck_arr_overall.append(1.0-selfcheck_scores_bert_overall[0]) #bert score flipped
        selfcheck_arr_average.append(1.0-selfcheck_scores_bert_average[0]) #bert score flipped
        selfcheck_ngram_average.append(1.0-np.exp(-selfcheck_ngram_overall[0]['doc_level']['avg_neg_logprob']))
        correct_arr.append(int(correct))
    except Exception as err:
        logger.error(f"Error at index {idx}: {err}")
        continue

logger.warning(f"Total samples processed: {len(correct_arr)}")
logger.warning(f"Total samples skipped: {len(dataset) - len(correct_arr)}")

roc_score = roc_auc_score(correct_arr, selfcheck_arr_overall)
logger.warning(f"AUROC for self check overall: {roc_score}")

roc_score = roc_auc_score(correct_arr, selfcheck_arr_average)
logger.warning(f"AUROC for self check average: {roc_score}")

roc_score = roc_auc_score(correct_arr, selfcheck_ngram_average)
logger.warning(f"AUROC for self check ngram: {roc_score}")

output_file = f"./logs/selfcheck/selfcheck_{model_name}_{dataset_name}_{gpu}.pickle"
with open(output_file, "wb") as outfile:
    outfile.write(pickle.dumps(selfcheck_dict))
logger.warning(f"Results saved to {output_file}")
logger.warning(f"Log file saved to {log_filename}")
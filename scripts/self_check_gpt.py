"""Config-driven SelfCheckGPT baseline."""

import argparse
import logging
import os
import pickle
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import spacy
import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore, SelfCheckNgram
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts import answer_trivia
from src.config import load_config
from src.loaders import load_dataset, load_llm


def configure_logger(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"selfcheck_log_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("self_check_gpt")
    logger.setLevel(logging.WARNING)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger, log_path


def generate_responses(
    question,
    reference_response,
    model,
    tokenizer,
    selfcheck_bertscore,
    selfcheck_ngram,
    nlp,
    trials,
    temperature,
    max_new_tokens,
    logger,
):
    """Sample responses and calculate SelfCheckGPT scores."""
    inputs = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
    start_pos = inputs.shape[-1]
    sampled_responses = []
    for _ in range(trials):
        outputs = model.generate(
            inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated_text = tokenizer.decode(
            outputs.sequences[0][start_pos:], skip_special_tokens=True
        ).replace("\n", " ").strip()
        if generated_text:
            sampled_responses.append(generated_text)

    if not sampled_responses:
        logger.warning("No valid responses generated for question: %s", question)
        return [], [], [], []

    overall_score = selfcheck_bertscore.predict(
        sentences=[reference_response],
        sampled_passages=sampled_responses,
    )[0]
    sentences = [
        sentence.text.strip()
        for sentence in nlp(reference_response).sents
        if len(sentence.text.strip()) > 3
    ]
    if sentences:
        average_score = statistics.mean(
            selfcheck_bertscore.predict(
                sentences=sentences,
                sampled_passages=sampled_responses,
            )
        )
    else:
        average_score = overall_score
    ngram_score = selfcheck_ngram.predict(
        sentences=sentences or [reference_response],
        passage=reference_response,
        sampled_passages=sampled_responses,
    )
    return sampled_responses, [overall_score], [average_score], [ngram_score]


def run(config):
    selfcheck_config = config["self_check"]
    model_config = config["model"]
    dataset_config = config["dataset"]
    logger, log_path = configure_logger(config["paths"]["log_dir"])
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    dataset_name = selfcheck_config.get("dataset", "trivia_qa")
    model_name = model_config["name"]
    device = torch.device(
        model_config.get("device", "cpu")
        if torch.cuda.is_available()
        else "cpu"
    )
    model, tokenizer = load_llm(
        model_name=model_name,
        cache_dir=model_config["cache_dir"],
        device="auto" if device.type == "cuda" else "cpu",
        dtype=model_config.get("dtype", "float16"),
        trust_remote_code=model_config.get("trust_remote_code", True),
    )
    dataset = load_dataset(
        dataset_name,
        Path(dataset_config["data_dir"]),
        start=dataset_config.get("start", 0),
        end=dataset_config.get("end", 10000),
        fava_samples=dataset_config.get("samples", 460),
    )
    if dataset_name != "trivia_qa":
        raise ValueError("SelfCheckGPT currently supports dataset.name='trivia_qa'.")

    selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
    selfcheck_ngram = SelfCheckNgram(n=selfcheck_config.get("ngram", 1))
    nlp = spacy.load(selfcheck_config.get("spacy_model", "en_core_web_sm"))
    trials = selfcheck_config.get("trials", 10)
    temperature = selfcheck_config.get("temperature", 0.1)
    max_new_tokens = selfcheck_config.get("max_new_tokens", 100)
    records = {key: [] for key in (
        "question", "response", "str_response", "start_pos", "correct",
        "hitemp_str_responses", "selfcheck_scores_bert_overall",
        "selfcheck_scores_bert_average", "selfcheck_ngram_overall",
    )}
    overall_scores, average_scores, ngram_scores, correct_labels = [], [], [], []

    for index, (question, answers) in enumerate(tqdm(dataset)):
        try:
            response, text, _, start_pos, correct, _ = answer_trivia(
                question, answers, model, tokenizer, model_name
            )
            sampled, overall, average, ngram = generate_responses(
                question,
                text,
                model,
                tokenizer,
                selfcheck_bertscore,
                selfcheck_ngram,
                nlp,
                trials,
                temperature,
                max_new_tokens,
                logger,
            )
            if not sampled:
                continue
            records["question"].append(question)
            records["response"].append(response)
            records["str_response"].append(text)
            records["start_pos"].append(start_pos)
            records["correct"].append(correct)
            records["hitemp_str_responses"].append(sampled)
            records["selfcheck_scores_bert_overall"].append(overall)
            records["selfcheck_scores_bert_average"].append(average)
            records["selfcheck_ngram_overall"].append(ngram)
            overall_scores.append(1.0 - overall[0])
            average_scores.append(1.0 - average[0])
            ngram_scores.append(1.0 - np.exp(-ngram[0]["doc_level"]["avg_neg_logprob"]))
            correct_labels.append(int(correct))
        except Exception as error:
            logger.error("Error at index %s: %s", index, error)

    logger.warning("Total samples processed: %s", len(correct_labels))
    logger.warning("Total samples skipped: %s", len(dataset) - len(correct_labels))
    metrics = {
        "overall": roc_auc_score(correct_labels, overall_scores),
        "average": roc_auc_score(correct_labels, average_scores),
        "ngram": roc_auc_score(correct_labels, ngram_scores),
    }
    output_dir = Path(selfcheck_config.get("output_dir", "./logs/selfcheck_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"selfcheck_{model_name.replace('/', '_')}_{dataset_name}_"
        f"{len(dataset)}samples.pickle"
    )
    with output_path.open("wb") as file:
        pickle.dump(records, file)
    for name, score in metrics.items():
        logger.warning("AUROC for self check %s: %s", name, score)
    logger.warning("Results saved to %s", output_path)
    logger.warning("Log file saved to %s", log_path)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

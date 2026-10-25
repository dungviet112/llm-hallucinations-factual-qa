from pathlib import Path
import torch
import json
import os
from collections import defaultdict
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = Path("../.cache/models")
model_dir.mkdir(parents=True, exist_ok=True)

class SingleMLP_Classifier(torch.nn.Module):
    def __init__(self, input_shape, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(256, 2)
            )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def get_classifier_input_dim(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    weight_key = "linear_relu_stack.0.weight"
    if weight_key not in checkpoint:
        raise KeyError(f"Missing {weight_key} in checkpoint: {checkpoint_path}")
    return checkpoint[weight_key].shape[1], checkpoint


def load_model_from_checkpoint(checkpoint_dir, model_name, feature, device=device):
    """
    Load the LLM model and MLP classifier from checkpoint for inference.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory containing saved model weights
        model_name: Name of the LLM model to load
        device: Device to load models on (cuda or cpu)
    
    Returns:
        tuple: (tokenizer, llm_model, classifier_model)
    """
    try:
        # Load the base LLM model and tokenizer
        model_path = f"meta-llama/{model_name}"
        print(f"Loading base LLM model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=model_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        llm_model.eval()
        
        # Initialize and load the MLP classifier
        print(f"Loading MLP classifier from checkpoint: {checkpoint_dir}")
        # Load the saved weights
        checkpoint_path = Path(checkpoint_dir) / model_name / f"mlp_{feature}.pt"
        if checkpoint_path.exists():
            input_shape, checkpoint = get_classifier_input_dim(checkpoint_path)
            classifier_model = SingleMLP_Classifier(input_shape=input_shape).to(device)
            classifier_model.load_state_dict(checkpoint)
            classifier_model.eval()
            print(f"Successfully loaded classifier from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print(f"Available checkpoints in {checkpoint_dir}:")
            for f in Path(checkpoint_dir).rglob("*.pt"):
                print(f"  - {f}")
            classifier_model = SingleMLP_Classifier(input_shape=llm_model.config.hidden_size).to(device)
        
        return tokenizer, llm_model, classifier_model
        
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}", file=sys.stderr)
        raise


STOP_TOKENS = {13, 128001, 128009}
def get_next_token(x, model):
    with torch.no_grad():
        outputs = model(x, output_hidden_states=True)
        return outputs.logits, outputs.hidden_states


def generate_response(x, model, *, max_length=50, pbar=False):
    response = []
    generated_embeddings = []
    bar = tqdm(range(max_length)) if pbar else range(max_length)
    for step in bar:
        logits, hidden_states = get_next_token(x, model)
        last_token_emb = hidden_states[-1][:, -1, :].squeeze(0).detach().to(torch.float32).cpu().numpy()
        generated_embeddings.append(last_token_emb)

        next_token = logits.squeeze()[-1].argmax()
        x = torch.concat([x, next_token.view(1, -1)], dim=1)
        response.append(next_token)
        if next_token in STOP_TOKENS and step > 5:
            break
    return torch.stack(response).cpu().numpy(), logits.squeeze(), generated_embeddings


def answer_question(question, model, tokenizer, *, max_length=50, pbar=False):
    input_ids = tokenizer(question, return_tensors='pt').input_ids.to(model.device)
    response, logits, generated_embeddings = generate_response(input_ids, model, max_length=max_length, pbar=pbar)
    str_response = tokenizer.decode(response, skip_special_tokens=True)
    return response, str_response, logits, input_ids.shape[-1], generated_embeddings


def infer_hallucination(text, tokenizer, llm_model, classifier_model, feature="context_emb", device=device):
    """
    Inference function to answer the given question then detect hallucination.
    
    Args:
        text: Input text to evaluate
        tokenizer: Tokenizer for the LLM
        llm_model: The base LLM model
        classifier_model: The trained MLP classifier
        device: Device to run inference on
    
    Returns:
        dict: Prediction results with label and confidence
    """
    classifier_model.eval()
    llm_model.eval()

    classifier_device = next(classifier_model.parameters()).device

    layer14_outputs = []
    layer14_handle = None
    if feature == "activation":
        target_module = llm_model.get_submodule("model.layers.14.mlp.up_proj")

        def capture_layer14_activation(module, inp, out):
            layer14_outputs.append(out.detach().to(torch.float32).cpu())

        layer14_handle = target_module.register_forward_hook(capture_layer14_activation)

    try:
        _, str_response, _, start_at, generated_embeddings = answer_question(text, llm_model, tokenizer)
    finally:
        if layer14_handle is not None:
            layer14_handle.remove()

    if feature == "activation":
        if not layer14_outputs:
            raise RuntimeError("Failed to collect layer 15 activation")
        pooled_embedding = layer14_outputs[-1][:, start_at, :].to(device=classifier_device)
    else:
        # Build the classifier input directly on the classifier's device to avoid CPU/CUDA mismatch.
        pooled_embedding = torch.tensor(np.stack(generated_embeddings), dtype=torch.float32, device=classifier_device)
        pooled_embedding = torch.mean(pooled_embedding, dim=0, keepdim=True)

    # Get prediction from classifier
    logits = classifier_model(pooled_embedding)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_label = int(torch.argmax(probabilities, dim=-1).item())
    confidence = probabilities[0, predicted_label].item()
    
    return {
        "response": str_response,
        "correct": predicted_label,  # 0: false (hallucinated), 1: correct (non-hallucinated)
        "confidence": confidence,
        "probabilities": {
            "non_hallucinated": probabilities[0, 1].item(),
            "hallucinated": probabilities[0, 0].item()
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Hallucination detection using trained MLP classifier")
    parser.add_argument("--question", "-q", type=str, help="Input text prompt as question")
    parser.add_argument(
        "--checkpoint-dir",
        "-c",
        default="./checkpoints",
        help="Path to checkpoint directory containing saved model weights"
    )
    parser.add_argument(
        "--model-name",
        "-m",
        default="Llama-3.1-8B-Instruct",
        help="Name of the LLM model to use for inference"
    )
    parser.add_argument("--feature", "-f", default="context_emb", choices=["context_emb", "activation"], help="Type of feature to use for classification")
    parser.add_argument("--json-output", "-j", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    # Get text input
    if not args.question:
        print("ERROR: --question argument is required", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(2)

    # Load model and checkpoint
    try:
        print("Loading models...", file=sys.stderr)
        tokenizer, llm_model, classifier_model = load_model_from_checkpoint(
            args.checkpoint_dir,
            model_name=args.model_name,
            feature=args.feature,
            device=device
        )
        print(f"Successfully loaded models on device: {device}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to load models: {e}", file=sys.stderr)
        sys.exit(1)

    # Run inference
    try:
        print("Running inference...", file=sys.stderr)
        result = infer_hallucination(args.question, tokenizer, llm_model, classifier_model, feature=args.feature, device=device)
        
        # Output results
        if args.json_output:
            with open("output.json", "w", encoding="utf-8") as f:
                output = {
                    "question": args.question,
                    "response": result["response"],
                    "correct": result["correct"],
                    "confidence": result["confidence"],
                    "hallucination_score": result["probabilities"]['hallucinated']
                }
                json.dump(output, f, ensure_ascii=False, indent=2)
                print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            label_name = "Hallucinated" if result["correct"] == 0 else "Non-hallucinated"
            print(f"Question: {args.question}")
            print(f"Response: {result['response']}")
            print(f"Correct: {label_name}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities:")
            print(f"  Non-hallucinated: {result['probabilities']['non_hallucinated']:.4f}")
            print(f"  Hallucinated: {result['probabilities']['hallucinated']:.4f}")
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
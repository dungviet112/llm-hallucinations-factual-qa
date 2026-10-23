## Install the required packages with Python 3.10+:
   ```bash
   pip install -r requirements.txt
   ```

## Command-Line Arguments

You can customize the script's behavior by passing the following arguments in the terminal:

| Argument | Short Flag | Type | Description | Required? | Default |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--question` | `-q` | *string* | The input text prompt or question to process. | **Yes** | *None* |
| `--checkpoint-dir` | `-c` | *string* | Path to the directory containing saved model weights. | No | `./checkpoints` |
| `--model-name` | `-m` | *string* | Name of the LLM model to use for inference. | No | `Llama-3.1-8B-Instruct` |
| `--feature` | `-f` | *string* | Type of feature to use for classification. <br>Options: `context_emb`, `activation`. | No | `context_emb` |
| `--json-output` | `-j` | *flag* | If passed, outputs the final results in JSON format. | No | *Disabled (False)* |

---

## How to Run the Script

### 1. Basic Example (Default Settings)
To run the script with the default Llama-3.1 model and default contextual embedding features, you only need to provide the `-q` argument:
```bash
python main.py -q "What is the capital of Germany?"
```

### 2. Advanced Example (Selected internal state & JSON Output)
To use a different feature type and save the result as a JSON:
```bash
python main.py \
  -q "What is the capital of Germany?" \
  -f activation \
  -j
```
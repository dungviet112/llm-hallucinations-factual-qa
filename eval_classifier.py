import numpy as np
import scipy as sp
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import random
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import *
from data_reader import load_hidden_states

gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
batch_size = 128
learning_rate = 1e-4
weight_decay = 1e-2
save_model_dir = "./models"
os.makedirs(save_model_dir, exist_ok=True)

# path to the results file containing model features and their corresponding labels (hallu/non-hallu)
results_file = "./results/open_llama_7b_trivia_qa_start-0_end-10000_6_16.pickle"

# initialize classifier
classifier = SingleMLP_Classifier

def gen_classifier_roc(inputs, labels, model, save_name="classifier_model.pt"):
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels.astype(int), test_size=0.2, random_state=123)
    classifier_model = model(X_train.shape[1]).to(device)
    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(torch.long).to(device)
    X_test = torch.tensor(X_test).to(device)
    y_test = torch.tensor(y_test).to(torch.long).to(device)

    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for _ in tqdm(range(1001)):
        optimizer.zero_grad()
        sample = torch.randperm(X_train.shape[0])[:batch_size]
        pred = classifier_model(X_train[sample])
        loss = torch.nn.functional.cross_entropy(pred, y_train[sample])
        loss.backward()
        optimizer.step()
    torch.save(classifier_model.state_dict(), f"{save_model_dir}/{save_name}")

    classifier_model.eval()
    with torch.no_grad():
        pred = torch.nn.functional.softmax(classifier_model(X_test), dim=1)
        prediction_classes = (pred[:,1]>0.5).type(torch.long).cpu()
        return roc_auc_score(y_test.cpu(), pred[:,1].cpu()), (prediction_classes.numpy()==y_test.cpu().numpy()).mean(), prediction_classes.numpy(), y_test.cpu().numpy()


def compute_confusion_stats(y_true, y_pred, labels=(0, 1)):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    tn, fp, fn, tp = cm.ravel()
    stats = {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": float((tp + tn) / cm.sum()),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
    }
    return stats, cm


def plot_confusion_matrix(cm, labels=("Hallucinated", "Non-hallucinated"), save_path="./results/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.set_facecolor("white")

    cmap = plt.cm.Blues
    img = ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(1, cm.max()))

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, color="#0f172a", fontsize=11)
    ax.set_yticklabels(labels, color="#0f172a", fontsize=11)
    ax.set_xlabel("Predicted label", color="#0f172a", fontsize=11)
    ax.set_ylabel("True label", color="#0f172a", fontsize=11)
    ax.set_title("Hallucination Classifier Confusion Matrix", color="#0f172a", fontsize=12, pad=10)
    ax.tick_params(axis="both", colors="#0f172a")
    for spine in ax.spines.values():
        spine.set_color("#1f2937")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_color = cmap(img.norm(cm[i, j]))
            brightness = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
            text_color = "white" if brightness < 0.55 else "#111827"
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#0f172a")
    plt.setp(cbar.ax.get_yticklabels(), color="#0f172a")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path

all_results = {}
try:
    classifier_results = {}
    print(f"Loading data from {results_file}")
    results, correct = load_hidden_states(results_file)

    if "trivia_qa" in results_file:
        # attributes
        X_train, X_test, y_train, y_test = train_test_split(results['attributes_first'], correct.astype(int), test_size=0.2, random_state=1234)
        rnn_model = RNN_Classifier().to(device)
        optimizer = torch.optim.AdamW(rnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for step in tqdm(range(1001)):
            x_sub, y_sub = zip(*random.sample(list(zip(X_train, y_train)), batch_size))
            y_sub = torch.tensor(y_sub).to(torch.long).to(device)
            optimizer.zero_grad()
            preds = torch.stack([rnn_model(torch.tensor(i).view(1, -1, 1).to(torch.float).to(device)) for i in x_sub])
            loss = torch.nn.functional.cross_entropy(preds, y_sub)
            loss.backward()
            optimizer.step()
        torch.save(rnn_model.state_dict(), f"{save_model_dir}/rnn_model.pt")
        
        preds = torch.stack([rnn_model(torch.tensor(i).view(1, -1, 1).to(torch.float).to(device)) for i in X_test])
        preds = torch.nn.functional.softmax(preds, dim=1)
        prediction_classes = (preds[:,1]>0.5).type(torch.long).cpu()
        classifier_results['attribution_rnn_roc'] = roc_auc_score(y_test, preds[:,1].detach().cpu().numpy())
        classifier_results['attribution_rnn_acc'] = (prediction_classes.numpy()==y_test).mean()

        # logits
        first_logits = np.stack([sp.special.softmax(i[j]) for i, j in zip(results['logits'], results['start_pos'])])
        first_logits_roc, first_logits_acc = gen_classifier_roc(first_logits, correct, model=classifier, save_name="mlp_logit.pt")
        classifier_results['first_logits_roc'] = first_logits_roc
        classifier_results['first_logits_acc'] = first_logits_acc

        # fully connected
        for layer in range(results['first_fully_connected'][0].shape[0]):
            layer_roc, layer_acc = gen_classifier_roc(np.stack([i[layer] for i in results['first_fully_connected']]), correct, model=classifier, save_name=f"mlp_fc_layer_{layer}.pt")
            classifier_results[f'first_fully_connected_roc_{layer}'] = layer_roc
            classifier_results[f'first_fully_connected_acc_{layer}'] = layer_acc

        # attention
        for layer in range(results['first_attention'][0].shape[0]):
            layer_roc, layer_acc = gen_classifier_roc(np.stack([i[layer] for i in results['first_attention']]), correct, model=classifier, save_name=f"mlp_attn_layer_{layer}.pt")
            classifier_results[f'first_attention_roc_{layer}'] = layer_roc
            classifier_results[f'first_attention_acc_{layer}'] = layer_acc
        
    # contextual embeddings
    pooled_embeddings = []
    for i in range(len(results['generated_embeddings'])):
        embeddings = results['generated_embeddings'][i]
        emb_sequence = torch.tensor(embeddings, dtype=torch.float32)
        # pooled_emb = torch.mean(emb_sequence, dim=0)  # mean pooling
        pooled_emb = torch.amax(emb_sequence, dim=0)    # max pooling
        pooled_embeddings.append(pooled_emb)

    pooled_embeddings_tensor = torch.stack(pooled_embeddings)
    # pooled_embeddings_tensor = np.stack(results['generated_embeddings'])
    context_emb_roc, context_emb_acc, prediction_classes, y_true = gen_classifier_roc(pooled_embeddings_tensor, correct, model=classifier, save_name="mlp_context_emb.pt")
    confusion_stats, confusion_matrix_values = compute_confusion_stats(y_true, prediction_classes, labels=(0, 1))
    confusion_plot_path = plot_confusion_matrix(
        confusion_matrix_values,
        labels=("Hallucinated", "Non-hallucinated"),
        # title="Confusion Matrix",
        save_path="./results/2100_samples_SATBench_Llama3.1-8B/confusion_matrix_hallu.png"
    )

    classifier_results['context_emb_roc'] = context_emb_roc
    classifier_results['context_emb_acc'] = context_emb_acc
    classifier_results['TP'] = confusion_stats['TP']
    classifier_results['TN'] = confusion_stats['TN']
    classifier_results['FP'] = confusion_stats['FP']
    classifier_results['FN'] = confusion_stats['FN']
    
    all_results[results_file] = classifier_results.copy()
except Exception as err:
    print(err)

for k,v in all_results.items():
    for kk, vv in v.items():
        print(f"{kk}: {vv}")
        
with open("./results/2100_samples_SATBench_Llama3.1-8B/score_result_max512_temp0.0_CoT_steps.txt", "a") as text_file:
    text_file.write(f"Results for {str(classifier)}:\n")
    for k,v in all_results.items():
        for kk, vv in v.items():
            text_file.write(str(kk) + " " + str(vv) + "\n")

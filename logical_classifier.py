import json
import os
import pickle
from pathlib import Path

import numpy as np
import scipy as sp
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import *

gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 128
learning_rate = 1e-4
weight_decay = 1e-2
save_model_dir = "./models"
os.makedirs(save_model_dir, exist_ok=True)

result_dir = Path("./results/10000_samples_HaluEval_Qwen2.5-7B-Instruct_v2")
embeddings_path = result_dir / "contextual_embs.pkl"
labels_path = result_dir / "labels.json"

classifier = SingleMLP_Classifier


def load_contextual_embeddings_by_id(embeddings_path=embeddings_path, labels_path=labels_path):
    with open(embeddings_path, "rb") as f:
        generated_embeddings, labels = pickle.load(f)

    # with open(labels_path, "r", encoding="utf-8") as f:
    #     label_metadata = json.load(f)

    # emb_by_id = {idx: np.asarray(embedding, dtype=np.float32) for idx, embedding in enumerate(embeddings)}
    # label_by_id = {}
    # labels = []
    # print(len(label_metadata))
    # print("emb:", len(embeddings))
    # for item in label_metadata:
    #     item_id = item.get("id")
    #     label = item.get("label")
    #     if item_id is None or label is None:
    #         continue
    #     labels.append(int(label))
    # # matched_ids = sorted(set(emb_by_id) & set(label_by_id))
    pooled_embeddings = []
    for i in range(len(generated_embeddings)):
        embeddings = generated_embeddings[i]
        emb_sequence = torch.tensor(embeddings, dtype=torch.float32)
        # pooled_emb = torch.mean(emb_sequence, dim=0)
        pooled_emb = torch.max(emb_sequence, dim=0).values
        pooled_embeddings.append(pooled_emb)
    pooled_embeddings = torch.stack(pooled_embeddings)

    X = pooled_embeddings
    y = labels

    return X, y


def gen_classifier_roc(inputs, labels, model, save_name="classifier_model.pt"):
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=123)
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
    # torch.save(classifier_model.state_dict(), f"{save_model_dir}/{save_name}")

    classifier_model.eval()
    with torch.no_grad():
        pred = torch.nn.functional.softmax(classifier_model(X_test), dim=1)
        prediction_classes = (pred[:, 1] > 0.5).type(torch.long).cpu()
        return roc_auc_score(y_test.cpu(), pred[:, 1].cpu()), (prediction_classes.numpy() == y_test.cpu().numpy()).mean()


def main():
    try:
        inputs, labels = load_contextual_embeddings_by_id()
        print(f"Loaded {len(inputs)} contextual embeddings and {len(labels)} labels from {labels_path}")

        context_emb_roc, context_emb_acc = gen_classifier_roc(inputs, labels, model=classifier, save_name="mlp_context_emb.pt")
        print(f"context_emb_roc: {context_emb_roc}")
        print(f"context_emb_acc: {context_emb_acc}")

        with open(result_dir / "score_result.txt", "a", encoding="utf-8") as text_file:
            text_file.write(f"context_emb_roc {context_emb_roc}\n")
            text_file.write(f"context_emb_acc {context_emb_acc}\n")
    except Exception as err:
        print(err)


if __name__ == "__main__":
    main()

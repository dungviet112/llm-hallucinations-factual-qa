import numpy as np
import scipy as sp
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random
from tqdm import tqdm

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
results_file = "./results/10000_samples_OpenLlama/open_llama_7b_trivia_qa_start-0_end-10000_6_16.pickle"

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
        return roc_auc_score(y_test.cpu(), pred[:,1].cpu()), (prediction_classes.numpy()==y_test.cpu().numpy()).mean()

all_results = {}
try:
    classifier_results = {}
    results, correct = load_hidden_states(results_file)

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
    first_logits_roc, first_logits_acc = gen_classifier_roc(first_logits, correct, model=SingleMLP_Classifier, save_name="mlp_logit.pt")
    classifier_results['first_logits_roc'] = first_logits_roc
    classifier_results['first_logits_acc'] = first_logits_acc

    # fully connected
    for layer in range(results['first_fully_connected'][0].shape[0]):
        layer_roc, layer_acc = gen_classifier_roc(np.stack([i[layer] for i in results['first_fully_connected']]), correct, model=SingleMLP_Classifier, save_name=f"mlp_fc_layer_{layer}.pt")
        classifier_results[f'first_fully_connected_roc_{layer}'] = layer_roc
        classifier_results[f'first_fully_connected_acc_{layer}'] = layer_acc

    # attention
    for layer in range(results['first_attention'][0].shape[0]):
        layer_roc, layer_acc = gen_classifier_roc(np.stack([i[layer] for i in results['first_attention']]), correct, model=SingleMLP_Classifier, save_name=f"mlp_attn_layer_{layer}.pt")
        classifier_results[f'first_attention_roc_{layer}'] = layer_roc
        classifier_results[f'first_attention_acc_{layer}'] = layer_acc
    
    # contextual embeddings
    pooled_embeddings = []
    for i in range(len(results['generated_embeddings'])):
        embeddings = results['generated_embeddings'][i]
        emb_sequence = torch.tensor(embeddings, dtype=torch.float32)
        pooled_emb = torch.mean(emb_sequence, dim=0)
        pooled_embeddings.append(pooled_emb)

    pooled_embeddings_tensor = torch.stack(pooled_embeddings)
    context_emb_roc, context_emb_acc = gen_classifier_roc(pooled_embeddings_tensor, correct, model=SingleMLP_Classifier, save_name="mlp_context_emb.pt")
    classifier_results['context_emb_roc'] = context_emb_roc
    classifier_results['context_emb_acc'] = context_emb_acc
    
    all_results[results_file] = classifier_results.copy()
except Exception as err:
    print(err)

for k,v in all_results.items():
    print(k, v)
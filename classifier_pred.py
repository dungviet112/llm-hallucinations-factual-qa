from model import *
import os
import torch
import numpy as np
from data_reader import load_hidden_states
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

classifier = SingleMLP_Classifier
results_file = "./results/2100_samples_SATBench_Llama3.1-8B/contextual_embeddings_max1024_temp0.0.pkl"
classifier_model = classifier(4096).to(device)
checkpoint = "./models/Llama-3.1-8B-Instruct/mlp_context_emb.pt"
state_dict = torch.load(checkpoint, map_location=device)
classifier_model.load_state_dict(state_dict)
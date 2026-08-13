# Identifying Hallucinations in LLMs

## Overview
This repo investigates the hallucination detection by treating the problem as binary classification over model artifacts, showing that internal representations can reveal whether an answer is likely to be factual before generation is fully complete.

## Setup

Set up the conda env by running `setup.sh`
```sh
bash setup.sh
```

## Data sources

There are 2 datasets: **TriviaQA** and **TREX**.

In particular, while **result_collector.py** uses **TriviaQA** directly, for TREX we do/save a sampling in the form of founders/capitals/place_of_birth.csv.
In the case of doing experiment on TREX, run `trex_parser.py` to create these data files first.
```sh
# trex
python trex_parser.py
```

## Artifact data collection with original hook function

Artifact data collection is done in **result_collector.py**.
For every sample, we gather some information including Q-A pair from dataset, response, artifacts of model (softmax probablities, feature attributions, self-attention, fully connected layer activations, and contextual embeddings), and label (hallucination/non-hallucination) then save them in pickle file.

Models/tokenizers are called from Huggingface. Softmax and contextual embeddings are collected directly from the model, attributions are collected using the integrated gradients (IG) method available in Captum and activations and attentions (model internal states) are collected using the **register_forward_hook** functionality.
```sh
python result_collector.py
```

<!-- ## Artifact data collection with pyvene

Artifacts includes activations, attention, softmax output, attributions.
Artifact data collection is done in **result_collector_pyvene.py** then is written in picke files.

Models/tokenizers are called from Huggingface. Softmax/logits are collected directly from the model, attributions are collected using the 
integrated gradients (IG) method available in Captum and activations and attentions are collected using the **pyvene** package which helps reduce the complexity of the code when collecting model internal states.
```sh
python result_collector_pyvene.py
``` -->

<!-- The experiment in **test_pyvene.ipynb** is about collecting artifacts from a sample QA after integrating with pyvene (the target is to check the type, shape of output) -->

## Classifiers

Training classifiers on IG, softmax, attention scores, FCC activations, contextual embeddings across the models/datasets. **model.py** consists of several different classifier architectures (Single layer MLP, multi-layer DNN + Residual block, and Multi-layer Transformer + Residual block) for the artifacts, then train and test them to get AUROC score with **eval_classifier.py** on the data collected by **result_collector.py**. The evaluated result (AUROC score) then be saved in **score_result.txt**
```sh
python eval_classifier.py
```

<!-- **Note**: The best performances are belonging to activations and attentions at last layer. -->

<!-- ## Plots

Data analysis (the plots in the paper) is done in **plots_tsne.ipynb** and **plots_entropy_and_pca.ipynb**. It corresponds to the 5.1 **Qualitative analysis** section of the paper, however most plots are collected in the appendix.

Once data is collected, we are iterested in comparative plots of softmax/IG attributions/activations across the models and datasets.
This is the reason why we collect the large dicts at the beginning of both notebooks. This is also a time consuming process, but note
that the notebook(s) can also be used on one model/dataset for fast experimentation.
Example: the data source directoiry (in our case **results**) would contain only capitals/falcon-40b_capitals_7_18.pickle while **founders**, **trivia**, **place_of_birth** stay empty. -->

## SelfCheckGPT

In this repo, selfcheckgpt is the baseline and compared to proposed method; a notebook is included. SelfcheckGPT does not perform well as the classifier, we hypothesize that this is because the models we use are small and the output for nonzero temperature is often subpar.
Selfcheckgpt uses the **bert-score** and **n-gram** methods from the its paper in **self_check_gpt.ipynb**.

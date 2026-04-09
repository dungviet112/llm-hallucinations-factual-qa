# Identifying Hallucinations in LLMs

## AUROC result on GPT-2

Up-to-date results at [here](https://docs.google.com/spreadsheets/d/13OxSkdCUJkjC2ns8Yjr0LK3VmrzOxTWAftfgcCyvR1k/edit?usp=sharing).

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

Classifiers and plots will be created on model/derived artifacts like activations, attention, softmax output, attributions.
Artifact data collection is done in **result_collector.py**, is **VERY** time consuming and best done on a powerful machine.
It will write picke files and it gathers more data than used in the paper (in the paper we look at last layer activations, etc).
Once acquired however, the same data can be used for a broader analysis if so desired.

We use models/tokenizers from Huggingface. Softmax/logits are collected directly from the model, attributions are collected using the 
integrated gradients (IG) method available in Captum and activations and attentions (model internal states) are collected using the **register_forward_hook** functionality.
```sh
python result_collector.py
```

## Artifact data collection with pyvene

Artifacts includes activations, attention, softmax output, attributions.
Artifact data collection is done in **result_collector_pyvene.py** then is written in picke files.

Models/tokenizers are called from Huggingface. Softmax/logits are collected directly from the model, attributions are collected using the 
integrated gradients (IG) method available in Captum and activations and attentions are collected using the **pyvene** package which helps reduce the complexity of the code when collecting model internal states.
```sh
python result_collector_pyvene.py
```

The experiment in **test_pyvene.ipynb** is about collecting artifacts from a sample QA after integrating with pyvene (the target is to check the type, shape of output)

## Classifiers

Training classifiers on IG, softmax, attention scores, FCC activations across the models/datasets. **classifier_model.ipynb** creates a RNN model for IG and single MLP models for the rest artifacts then trains them on the data collected by **result_collector.py** or **result_collector_pyvene**. The results are in tables 2 and 3 in the **Results** section of the paper.

**Note**: The best performances are belonging to activations and attentions at last layer.

## Plots

Data analysis (the plots in the paper) is done in **plots_tsne.ipynb** and **plots_entropy_and_pca.ipynb**. It corresponds to the 5.1 **Qualitative analysis** section of the paper, however most plots are collected in the appendix.

Once data is collected, we are iterested in comparative plots of softmax/IG attributions/activations across the models and datasets.
This is the reason why we collect the large dicts at the beginning of both notebooks. This is also a time consuming process, but note
that the notebook(s) can also be used on one model/dataset for fast experimentation.
Example: the data source directoiry (in our case **results**) would contain only capitals/falcon-40b_capitals_7_18.pickle while **founders**, **trivia**, **place_of_birth** stay empty.

## SelfCheckGPT

We try to use selfcheckgpt and compare to our results; a notebook is included. SelfcheckGPT does not perform well
with our models; we hypothesize that this is because the models we use are small and the output for nonzero temperature is often subpar. 
We use the **bert-score** and **n-gram** methods from the selfcheckgpt paper in **self_check_gpt.ipynb** and we report the results in the appendix **B** (additional results) of the paper.

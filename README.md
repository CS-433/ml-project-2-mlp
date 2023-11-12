# Enhancing Homepage Topic Classification via LLM-augmented Datasets (WIP)

Rough sketch of project outline:

The project is motivated by the following limitations in the original paper:
1. The Curlie dataset that was used for training contains only a single label per webpage. However, the human annotated data has shown that in fact multiple topics are often relevant for a single page. Thus, the model is penalised for possibly correct predictions during training.
2. The distribution of topic labels is imbalanced, e.g. Kids & Teens only accounts for ~1% of all pages. The downstream performance in these classes is generally lower.

Project idea:
We hypothesise that fine-tuning on a balanced, multi-labelled dataset improves multi-label performance, especially in the low-resource classes. To test this hypothesis, we aim to fine-tune the original Homepage2Vec model on the small existing crowd-sourced multi-labelled dataset in a first step. Next, we want to investigate if an LLM-generated or LLM-augmented fine-tuning dataset can reach similar dataset quality and can therefore be used as a replacement for the crowd-sourced fine-tuning dataset. If it proves a viable option, interesting follow-up research question arise, like how scaling up the fine-tuning dataset effects the downstream performance.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)

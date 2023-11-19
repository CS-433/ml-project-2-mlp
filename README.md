# Enhancing Homepage Topic Classification via LLM-augmented Datasets (WIP)

This project is being developed in collaboration with the [DLab](https://dlab.epfl.ch/) at EPFL as part of the [Machine Learning](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) (CS-433) course. The project aims to explore the use of [Language Model Augmentation](https://arxiv.org/abs/2105.03075) (LLM) to improve the classification accuracy in low-resource webpage topics. The project is based on the paper [Homepage2Vec:  Language-Agnostic Website Embedding and Classification](https://arxiv.org/pdf/2201.03677.pdf) and the resulting library [Homepage2Vec](https://github.com/epfl-dlab/homepage2vec)

## Project Outline

The project is motivated by the following limitations in the original Homepage2Vec paper:

1. The Curlie dataset that was used for training contains mostly a single label per webpage. However, multiple topics are often relevant for a single page. Thus, the model is penalised for possibly correct predictions during training.

2. The distribution of topic labels is imbalanced, e.g. Kids & Teens only accounts for ~1% of all pages. The downstream performance in these classes is generally lower.

This yields the following research questions:

We hypothesise that fine-tuning on a balanced, multi-labeled dataset improves multi-label performance, especially in the low-resource classes. To test this hypothesis, we aim to fine-tune the original Homepage2Vec model on the small existing crowd-sourced multi-labelled dataset in the first step. Next, we want to investigate if an LLM-generated or LLM-augmented fine-tuning dataset can reach similar dataset quality and can therefore be used as a replacement for the crowd-sourced fine-tuning dataset. If it proves a viable option, interesting follow-up research questions arise, like how scaling up the fine-tuning dataset affects the downstream performance.

## Reproducibility

To reproduce all results, this notebook should be run with the correct **Python version** inside the specified **virtual environment** to use all packages with the correct version.

We use [Poetry](https://python-poetry.org/) for package management and versioning. This project was developed for Python `3.10.13` and Poetry `1.2`. We recommend installing Python via [pyenv](https://github.com/pyenv/pyenv) and Poetry via [pipx](https://pypa.github.io/pipx/).

```bash
pyenv install 3.10
```

Then, install Poetry via `pipx`

```bash
pipx install poetry==1.2.0
```

The project has a `.python-version` in the root directory, which will automatically activate the correct Python version when you enter the project directory. You can check that the correct Python version is used by running `python --version` (should be Python `3.10.13`) and that `poetry --version` is `1.2.0`.

Next, we install all dependencies via Poetry:

```bash
poetry install
```

You can now run the project by using the virtual environment created by Poetry. You can choose to run individual scripts from the command line by prefixing the command with `poetry run`, e.g. `poetry run python main.py` or create a shell that will load the virtual environment via `poetry shell`. If you run the notebooks, make sure to select the correct kernel (should be `venv`).

To test if the setup was successful, you can run the unit tests via `poetry run pytest tests/test_setup.py`. It tests for the correct Python version and that all dependencies are installed.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)

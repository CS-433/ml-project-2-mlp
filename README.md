# 🌐 Enhancing Homepage Topic Classification via LLM-augmented Datasets (WIP)

This project is being developed in collaboration with the [DLab](https://dlab.epfl.ch/) at EPFL as part of the [Machine Learning](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) (CS-433) course. The project aims to explore the use of [Language Model Augmentation](https://arxiv.org/abs/2105.03075) (LLM) to improve the classification accuracy in low-resource webpage topics. The project is based on the paper [Homepage2Vec:  Language-Agnostic Website Embedding and Classification](https://arxiv.org/pdf/2201.03677.pdf) and the resulting library [Homepage2Vec](https://github.com/epfl-dlab/homepage2vec).

## 🎯 Project Outline

The project is motivated by the following limitations in the original Homepage2Vec paper:

1. The Curlie dataset that was used for training contains mostly a single label per webpage. However, multiple topics are often relevant for a single page. Thus, the model is penalised for possibly correct predictions during training.

2. The distribution of topic labels is imbalanced, e.g. Kids & Teens only accounts for ~1% of all pages. The downstream performance in these classes is generally lower.

This yields the following research questions:

We hypothesise that fine-tuning on a balanced, multi-labeled dataset improves multi-label performance, especially in the low-resource classes. To test this hypothesis, we aim to fine-tune the original Homepage2Vec model on the small existing crowd-sourced multi-labelled dataset in the first step. Next, we want to investigate if an LLM-generated or LLM-augmented fine-tuning dataset can reach similar dataset quality and can therefore be used as a replacement for the crowd-sourced fine-tuning dataset. If it proves a viable option, interesting follow-up research questions arise, like how scaling up the fine-tuning dataset affects the downstream performance.

## 🔁 Reproducibility

### ⚙️ Environment Setup
To reproduce all results, this notebook should be run with the correct **Python version** inside the specified **virtual environment** to use all packages with the correct version.

We use [Poetry](https://python-poetry.org/) for package management and versioning. This project was developed for Python `3.10.13` and Poetry `1.7`. We recommend installing Python via [pyenv](https://github.com/pyenv/pyenv) and Poetry via [pipx](https://pypa.github.io/pipx/).

```bash
pyenv install 3.10.13
```

Then, install Poetry via `pipx`

```bash
pipx install poetry==1.7.0
```

The project has a `.python-version` in the root directory, which will automatically activate the correct Python version when you enter the project directory. You can check that the correct Python version is used by running `python --version` (should be Python `3.10.13`) and that `poetry --version` is `1.7.0`.

Next, we install all dependencies via Poetry:

```bash
poetry install
```

You can now run the project by using the virtual environment created by Poetry. You can choose to run individual scripts from the command line by prefixing the command with `poetry run`, e.g. `poetry run python main.py` or create a shell that will load the virtual environment via `poetry shell`. If you run the notebooks, make sure to select the correct kernel (should be `venv`).

To test if the setup was successful, you can run the unit tests via `poetry run pytest tests/test_setup.py`. It tests for the correct Python version and that all dependencies are installed.

Last but not the least, since we are using OpenAI's API, create a `.env` file in the root directory and add your API key as follows:

```bash
OPENAI_API_KEY=<your-api-key>
```

Make sure that you have at least a few dollars so you do not run into any rate-limiting issues. From our experience, to label one of the provided websites on average costs around `$0.0001`. 


### 🧪 Run the Experiments

In our project, we define an experiment as a combination of a model, dataset and the labeler. We precisely define each of these parameters via [hydra](https?//hydra.cc/). The following table shows the available options for the parameters:

| Parameter | Description                                             | Available Options                 |
|-----------|---------------------------------------------------------|-----------------------------------|
| data   | Dataset to be labeled and then finetuned on            | `original`, `ours`                    |
| labeler   | Labeler you want to use for the dataset annotation | `human` (only available for `original` dataset), `gpt-labeler1`, `pt-labeler2`, `gpt-labeler3` |

To inspect more closely what hyper-parameters each of the options represent, you can look at the corresponding hydra config files: [model](conf/model/homepage2vec.yaml), [data](conf/data/) and [labeler](conf/labeler/). You can then either change the hyper-parameters directly in the
configuration file or via command line. For instance, to change the learning rate of an optimiser, you can put as argument to the command
`model.optimizer.lr=0.01`. Finally, here is an example of how to run an experiment:

```bash
poetry run train data=original labeler=human
```

The command then executes the entire pipeline:

1. Given the URLs associated with the given dataset, scrape the HTML content of the websites and save it to a disk.
2. Preprocess the HTML content and save it to a disk. Preprocessing extracts from the given URL and HTML content the following information: *top-level-domain*, *domain*, *title*, *description*, *metatags*, *keywords*, *links* and *text*. This step is neccessary for the labeler later on since this will determine the input context to the labeler.
3. Embed HTML and URL content using the pretrained [multi-lingual sentence transformer model](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1). This step is needed for the Homepage2Vec model fine-tuning.
4. Label the dataset using the given labeler. The labeler will use the preprocessed HTML content as input context and will then generate a label for each website. On a high level, we contruct a prompt that consist of the description of the labelling task including the possible 
[categories](data/meta/categories.txt) as well as [example website](data/meta/example-website.json) and [example-output](data/meta/example-labels.json). Finally, we add to the prompt the preprocessed HTML content of the website. We vary the amount of context we provide to each labeler, therefore the naming `gpt-labeler1`, `gpt-labeler2` etc. The higher the number, the more context we provide to the labeler. Importantly, `gpt-labeler2` has all the context of the labeler `gpt-labeler1` and so on. The labeler will then make a multi-label prediction for each website. The labels are then saved to a disk.
5. Finally, in the final step, we split the labeled dataset into a training and testing, fine-tune the Homepage2Vec model on the training set and evaluate the performance on the testing set. The results of the experiment with all hyper-parameters are then save to [wandb](https://wandb.ai/).
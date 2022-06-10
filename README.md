
#  Nested Named Entity Recognition in the Chilean Waiting List corpus.
Source code for the paper: Automatic extraction of nested entities in clinical referrals in Spanish. This repository contains brat to conll transformation and MSEN code.

## Install

1. Create an enviroment: `python -m venv venv` and activate it.
2. Run `pip install -r requirements.txt` to install all dependencies
3. Download the statistical model to perform the tokenization: `python -m spacy download es_core_news_lg`
4. In case you use a GPU NVIDIA RTX 3090, then install this PyTorch version: `pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html`

## Data

We release the CoNLL files already formatted for the MSEN model for simplicity. However, if you want to reproduce the transformation process, follow the steps in the Files section.

The files associated with each type of entity are located in the folder src/wl_files. These files correspond to 5000 annotations of the Chilean Waiting List transformed to the traditional CoNLL file format.

## Embeddings

Put the `cwlce.vec` embeddings in src/embeddings (it can be downloaded from here: https://zenodo.org/record/3924799).

The BERT and Flair contextual embeddings are generated using the Flair framework: https://github.com/zalandoresearch/flair. 

The selection of embeddings to be used can be modified in the `config.yaml` file.

## MSEN Training.

Training parameters can be changed in `config.yaml` file

Run the script `src/main.py`. The results will be printed to console, the models will be saved in models folder.

## Metrics

To obtain the standard nested NER metric of the model considering all entity types outputs, install the library nestednereval `pip install git+https://github.com/matirojasg/nestednereval.git` and then execute the script `metrics`.

## Files

ToDO


#  The Chilean Waiting List Corpus.
Code for the paper Automatic extraction of nested entities in clinical referrals in Spanish. This repository contains brat to conll transformation and Single Entity Model.

## Install

1. Create an enviroment: `python -m venv venv` and activate it.
2. Run `pip install -r requirements.txt` to install all dependencies
3. Download the statistical model to perform the tokenization: `python -m spacy download es_core_news_lg`
4. In case you use a GPU, then install this PyTorch version: `pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`


## Files

Multiple Single Entity architecture files:

1. Request credentials from authors to obtain data from the MINIO object repository. Write to the following email: pln@cmm.uchile.cl
2. Execute the following code in the data directory by entering the credentials:

`python msen_data.py --server minio.cmm.uchile.cl --access_key your_username --secret_key your_pass --types Disease Finding Abbreviation Medication Body_Part Family_Member Procedure`. Files will be located in the wl_files folder.

The input format is a CoNLL format, with one token per line, sentences
delimited by empty line. For each token, columns are separated by spaces. First
column is the surface token, second column is the label.

Neural Layered architecture files:

1. After having obtained the files for the WL Corpus, run the command: `python neural_layered_data.py`,  to obtain Neural Layered files in neural_layered_files. Files will be located in the neural_layered_files folder.

## Embeddings

Put the `cwlce.vec` embeddings (it can be downloaded from here: https://zenodo.org/record/3924799).

The BERT and Flair contextual embeddings are generated using the Flair framework: https://github.com/zalandoresearch/flair. 

The selection of embeddings to be used can be modified in the `params.json` file.

## MSEN Training.

Training parameters can be changed in `params.json` file

Run the script `train.py`. The results will be will be printed to console, the models will be saved in models folder.

The models will be stored in the models folder.

## Neural Layered experiments

We use the original repository of the paper: https://github.com/meizhiju/layered-bilstm-crf using files obtained in this repository. 

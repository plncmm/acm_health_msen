from dataset import NERCorpus
from models import NERTagger
from embeddings import Embeddings
from trainers import NERTrainer
import yaml
import torch 
import flair
import os

if __name__=='__main__':
    # Read configuration file
    with open('../config.yaml') as file:
        config = yaml.safe_load(file)

    available_gpu = torch.cuda.is_available()
    if available_gpu:
        print(f'GPU is available: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        flair.device = torch.device('cuda')
    else:
        flair.device = torch.device('cpu')


    seed = config['seed']

    flair.set_seed(seed)
    torch.cuda.empty_cache()

    
    actual_path = os.getcwd()
    directory = os.fsencode(f"{config['data_folder']}/")

    for file in os.listdir(directory):
        entity_type = os.fsdecode(file)
        corpus = NERCorpus(config['data_folder'], entity_type).create_corpus()
        tag_dictionary = corpus.make_label_dictionary(label_type = 'ner')
        embeddings = Embeddings(config).create_embeddings()
        tagger = NERTagger(embeddings = embeddings, tag_dictionary = tag_dictionary, config = config).create_tagger()
        trainer = NERTrainer(corpus = corpus, tagger = tagger, entity_type = entity_type, config = config).train()

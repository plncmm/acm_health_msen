from flair.trainers import ModelTrainer
from torch.optim import Adam, SGD

class NERTrainer:
    def __init__(self, corpus, tagger, config):  
        self.corpus = corpus
        self.tagger = tagger
        self.config = config

    def train(self):

        trainer: ModelTrainer = ModelTrainer(model = self.tagger, corpus = self.corpus)

        print(trainer)
        
        trainer.train(
            base_path = self.config['output_path'],
            learning_rate = self.config['learning_rate'],
            train_with_dev = self.config['train_with_dev'],
            train_with_test = self.config['train_with_test'],
            mini_batch_size = self.config['mini_batch_size'],
            max_epochs = self.config['max_epochs'],
            optimizer = eval(self.config['optimizer'])
            embeddings_storage_mode = 'none',
            ) 
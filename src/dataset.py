from flair.data import Corpus
from flair.datasets import ColumnCorpus

class NERCorpus:
    def __init__(self, data_folder, entity_type) -> None:
        self.data_folder = data_folder
        self.entity_type = entity_type
        
    def create_corpus(self) -> Corpus:
        corpus: Corpus = ColumnCorpus(data_folder = '{}/{}/'.format(self.data_folder, self.entity_type),  
                                                column_format = {0: 'text', 1: 'ner'},
                                                train_file = '{}_train.conll'.format(self.entity_type),
                                                test_file = '{}_test.conll'.format(self.entity_type),
                                                dev_file = '{}_dev.conll'.format(self.entity_type))
        return corpus
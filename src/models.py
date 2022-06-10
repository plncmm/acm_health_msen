from flair.models import SequenceTagger

class NERTagger:
    def __init__(self, embeddings, tag_dictionary, config) -> None:
        self.embeddings = embeddings
        self.tag_dictionary = tag_dictionary
        self.config = config

        
        
    def create_tagger(self) -> SequenceTagger:
        tagger: SequenceTagger = SequenceTagger(
                                    embeddings = self.embeddings,
                                    tag_dictionary = self.tag_dictionary,
                                    rnn_type = self.config['encoder'],
                                    hidden_size = self.config['hidden_size'],
                                    use_crf = self.config['use_crf'],
                                    tag_type = 'ner'
                                )
        return tagger
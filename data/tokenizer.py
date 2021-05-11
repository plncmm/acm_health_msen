from collections import OrderedDict
import codecs 
import spacy 
import re

TOKENIZATION_REGEXS = OrderedDict([
    # NERsuite-like tokenization: alnum sequences preserved as single
    # tokens, rest are single-character tokens.
    ('default', re.compile(r'([^\W_]+|.)')),
    # Finer-grained tokenization: also split alphabetical from numeric.
    ('fine', re.compile(r'([0-9]+|[^\W0-9_]+|.)')),
    # Whitespace tokenization
    ('space', re.compile(r'(\S+)')),
])


def tokenize_with_spacy(text, entities, tokenizer, lower_tokens=False, no_accent_marks=False, referral_name=None):
    """ 
    The given text is tokenized prioritizing not to lose entities that ends in the middle of a word.
    This because on many occasions words are stick together in a free text.
    """ 
    idx = 0
    no_tagged_tokens_positions = []
    tagged_tokens_positions = [(entity['start_idx'], entity['end_idx']) for entity in entities]
    entity_tokens = spacy_tokens(text, tagged_tokens_positions, tokenizer, lower_tokens, no_accent_marks)  
    for tagged_token in tagged_tokens_positions:
        no_tagged_tokens_positions.append((idx, tagged_token[0])) # We add text before tagged token
        idx = tagged_token[1] 
    no_tagged_tokens_positions.append((idx, len(text)))           # We add text from last token tagged end possition to end of text.
    no_entity_tokens = spacy_tokens(text, no_tagged_tokens_positions, tokenizer, lower_tokens, no_accent_marks)
    tokens = sorted(entity_tokens+no_entity_tokens, key=lambda entity:entity["start_idx"])
    return [tokens]

def spacy_tokens(text, pos_list, tokenizer, lower_tokens, no_accent_marks): 
    """ 
    Given a list of pairs of start-end positions in the text, 
    the text within these positions is tokenized and returned in tokens array.
    """
    tokens = []
     # TODO: Add new tokenizers (e.g, Nltk) to compare performance.
    for poss in pos_list:
        text_tokenized = tokenizer(text[poss[0]:poss[1]])
        for span in text_tokenized.sents:
            sentence = [text_tokenized[i] for i in range(span.start, span.end)]
            for token in sentence:
                token_dict = {}
                token_dict['start_idx'] = token.idx + poss[0]
                token_dict['end_idx'] = token.idx + poss[0] + len(token)
                token_dict['text'] = text[token_dict['start_idx']:token_dict['end_idx']]
                if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                    continue
                if len(token_dict['text'].split(' ')) != 1:
                    token_dict['text'] = token_dict['text'].replace(' ', '-')
                # TODO: Before adding token to token list, process irregular tokens with custom parsing.
                if lower_tokens: token_dict['text'] = token_dict['text'].lower()
                if no_accent_marks: token_dict = remove_accent_mark(token_dict)
                tokens.append(token_dict)
    
    return tokens


def remove_accent_mark(token_dict):
    try:
        token_dict['text'] = token_dict['text'].replace('á','a')
        token_dict['text'] = token_dict['text'].replace('é','e')
        token_dict['text'] = token_dict['text'].replace('í','i')
        token_dict['text'] = token_dict['text'].replace('ó','o')
        token_dict['text'] = token_dict['text'].replace('ú','u')
        return token_dict
    except:
        return token_dict

def fix_tokens(path, output_path):
    tokenization_re = TOKENIZATION_REGEXS.get('default')
    f = codecs.open(path, 'r', 'UTF-8').read()
    out = codecs.open(output_path, 'w', 'UTF-8')
    for line in f.split('\n'):
        if line!='':
            original_entities = line.split()[1:]
            new_tokens = [t for t in tokenization_re.split(line.split()[0]) if t]
            for i, token in enumerate(new_tokens):
                if i==0:
                    out.write(f"{token} {' '.join(original_entities)}\n")
                else:
                    new_entities = []
                    for entity in original_entities:
                        if entity[0]=='B':
                            new_entities.append(f'I-{entity[2:]}')
                        else:
                            new_entities.append(entity)
                    out.write(f"{token} {' '.join(new_entities)}\n")

        else:
            out.write(line+'\n')
    out.close()
    pass
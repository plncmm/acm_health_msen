import codecs 
import math 
import spacy 
import time 
from tqdm import tqdm
from entities import get_nested_entities, get_flat_entities
from tokenizer import tokenize_with_spacy


def convert_to_conll(referrals, annotations, entity_types, output_path):
    """ 
    Function used to create conll file format from ann-txt annotations.
    """
    output_file = codecs.open(output_path, 'w', 'UTF-8')

    tokenizer = spacy.load('es_core_news_lg', disable = ['ner', 'tagger'])

    for k, (referral, annotation) in tqdm(enumerate(zip(referrals, annotations))):
        nested_entities = get_nested_entities(annotation[1], referral, entity_types)
        nested_entities = sorted(nested_entities, key = lambda entity: entity["start_idx"])
        flat_entities = get_flat_entities(nested_entities, referral)
        flat_entities = sorted(flat_entities, key = lambda entity: entity["start_idx"])
        sentences = tokenize_with_spacy(referral[1], flat_entities, tokenizer)

        for sentence in sentences:
            inside_entity = {}
            for entity_type in entity_types:
                inside_entity[entity_type] = 0
            for i, token in enumerate(sentence):
                token['label'] = 'O'
                token_labels = []
                for entity in nested_entities:
                    
                    if token['start_idx'] < entity['start_idx']:
                        continue
                    
                    elif token['end_idx'] == entity['end_idx'] and token['start_idx'] == entity['start_idx'] or \
                        (token['start_idx']==entity['start_idx'] and token['text']==' '.join(entity['text'].split())) \
                        or (token['end_idx']==entity['end_idx'] and token['text']==' '.join(entity['text'].split())):

                        inside_entity[entity['label']] = 0
                        token_labels.append('B-' + entity['label'])

                    elif token['end_idx'] < entity['end_idx'] and not inside_entity[entity['label']]:
                        inside_entity[entity['label']] = 1
                        token_labels.append('B-' + entity['label'])

                    elif token['end_idx'] < entity['end_idx'] and inside_entity[entity['label']]:
                        token_labels.append('I-' + entity['label'])

                    elif token['end_idx'] == entity['end_idx'] and not inside_entity[entity['label']]:
                        inside_entity[entity['label']]=0
                        token_labels.append('B-' + entity['label'])
                        
                    elif token['end_idx'] == entity['end_idx'] and inside_entity[entity['label']]:
                        inside_entity[entity['label']]=0
                        token_labels.append('I-' + entity['label'])

                    elif entity['start_idx']>token['end_idx']:
                        break
                    else: 
                        continue

                if len(token_labels)!=0:
                    token_labels = sorted(token_labels, key=lambda entity:entity, reverse=True) 
                    if k==len(referrals)-1 and i==len(sentence)-1: output_file.write(f"{token['text']} {' '.join(token_labels)}")
                    else: output_file.write(f"{token['text']} {' '.join(token_labels)}\n")

                elif token['text']=='.' and i!=len(sentence)-1:
                    output_file.write(f"{token['text']} {token['label']}\n\n")

                else:
                     if k==len(referrals)-1 and i==len(sentence)-1: output_file.write(f"{token['text']} {token['label']}")
                     else: output_file.write(f"{token['text']} {token['label']}\n")
            
            
            if k!=len(referrals)-1: output_file.write('\n')

    output_file.close()



def create_partitions(filepath, train_path, dev_path, test_path):
    text = codecs.open(filepath, 'r', 'UTF-8').read()
    annotations = text.split('\n\n')
    n_examples = len(annotations)
    n_train = math.floor(n_examples*0.81)
    n_val =  math.floor(n_examples*0.09)
    n_test=  math.floor(n_examples*0.10)
    train = codecs.open(train_path, 'w', 'UTF-8')
    for i in range(0, n_train):
        if i!=n_train-1: train.write(annotations[i] + "\n\n")
        else: train.write(annotations[i])
    train.close() 

    dev = codecs.open(dev_path, 'w', 'UTF-8')
    for i in range(n_train, n_train+n_val):
        if i!=n_train+n_val-1: dev.write(annotations[i] +"\n\n")
        else: dev.write(annotations[i])
    dev.close()

    test = codecs.open(test_path, 'w', 'UTF-8')
    for i in range(n_train+n_val, n_examples):
        if i!=n_examples-1: test.write(annotations[i] +"\n\n")  
        else: test.write(annotations[i])  
    test.close() 

def create_single_entity_data(path):
    entities = ['Finding', 'Procedure', 'Disease', 'Body_Part', 'Abbreviation', 'Family_Member', 'Medication']
    text = codecs.open(path, 'r', 'UTF-8').read()
    
    for entity in entities:
        start = time.time()
        f_out = codecs.open(f'wl_files/{entity}/{entity}.conll', 'w', 'utf-8')
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line!='':
                original_entities = line.split()[1:]
                new_entities = []
                for val in original_entities:
                    if val in (f'B-{entity}',f'I-{entity}','O'):
                        new_entities.append(val)
                if len(new_entities)==0:
                    new_entities.append('O')
                f_out.write(f"{line.split()[0]} {' '.join(new_entities)}\n")
            else:
                if i!=len(lines)-1: f_out.write(line+'\n')
        f_out.close()
        create_partitions(f'wl_files/{entity}/{entity}.conll', f'wl_files/{entity}/{entity}_train.conll', f'wl_files/{entity}/{entity}_dev.conll', f'wl_files/{entity}/{entity}_test.conll')

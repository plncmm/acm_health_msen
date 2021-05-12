import codecs 
import time
import os
from brat_to_conll import create_partitions
import logging
logging.basicConfig(level=logging.INFO)

def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """

    def _validate_chunk(chunk, suffix):
        if chunk in ['O', 'B', 'I', 'E', 'S']:
            return

        if suffix:
            if not chunk.endswith(('-B', '-I', '-E', '-S')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

        else:
            if not chunk.startswith(('B-', 'I-', 'E-', 'S-')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks

def get_labels_from_conll(conll):
    annotations = conll.split('\n\n')
    entities_dict = {'Finding': [], 'Procedure': [], 'Disease': [], 'Body_Part': [], 'Abbreviation': [], 'Family_Member': [], 'Medication': []}
    print(f'Number of sentences: {len(annotations)}')
    for anno in annotations: 
      for line in anno.splitlines():
          line_info = line.split()
          for k, v in entities_dict.items():
              if f'B-{k}' in line_info[1:]:
                  entities_dict[k].append(f'B-{k}')
              elif f'I-{k}' in line_info[1:]: 
                  entities_dict[k].append(f'I-{k}')
              else:
                  entities_dict[k].append(f'O')
    return entities_dict

def get_all_entities(entities_dict):
    flatten_entities = []
    for k, v in entities_dict.items():
        entities = get_entities(v)
        flatten_entities+=entities
    return flatten_entities


def get_grade(entities_flatten):
    ar = []
    for index, entity_1 in enumerate(entities_flatten):
        if index == 0: start = time.time()
        grade = 1
        entity_1_inner_entities = []

        for entity_2 in entities_flatten:
            if entity_2 != entity_1 and entity_1[1] <= entity_2[1] and entity_1[2]>=entity_2[2]:
                if grade==1: grade+=1
                entity_1_inner_entities.append(entity_2)

        for inner_entities_1 in entity_1_inner_entities:
            entity_1_doble_inner_entities = []
            for entity_3 in entities_flatten:
                if entity_3!=entity_1 and entity_3!=inner_entities_1 and inner_entities_1[1]<=entity_3[1] and inner_entities_1[2]>=entity_3[2]:
                    if grade==2: grade+=1
                    entity_1_doble_inner_entities.append(entity_3)
            
            for doble_inner_entities_1 in entity_1_doble_inner_entities:
                for entity_4 in entities_flatten:
                    if entity_4!=entity_1 and entity_4!=inner_entities_1 and entity_4!=doble_inner_entities_1 and doble_inner_entities_1[1]<=entity_4[1] and doble_inner_entities_1[2]>=entity_4[2]:
                        if grade==3: grade+=1

        ar.append((entity_1[0], entity_1[1], entity_1[2], grade))

    return ar 

if __name__=='__main__':
    text = codecs.open('../wl_files/wl_entities.conll', 'r', 'utf-8').read()
    entities_dict = get_labels_from_conll(text)
    entities_flatten = get_all_entities(entities_dict)
    logging.info('Obtaining the depth-level of each entity...')
    ar = get_grade(entities_flatten)
    text = codecs.open('../wl_files/wl_entities.conll', 'r', 'UTF-8').read()
    if not os.path.exists('../neural_layered_files'): os.mkdir('../neural_layered_files') 
    new_text = codecs.open('../neural_layered_files/entities.conll', 'w', 'UTF-8')
    annotations = text.split('\n\n')[:-1]
    entities_index = 0
    count = 0
    logging.info('Creating Neural Layered files...')
    for k, anno in enumerate(annotations):
        count+=1
        if count%500==0: print(f"{count} sentences transformed")
        lines = anno.splitlines()
        for i, line in enumerate(lines):
            token = line.split()[0]
            entities = ['O']*4
            
            for entity in ar:
            
                if entities_index>=entity[1] and entities_index<=entity[2]:
                    added = 0
                    grade = entity[3]
                    if entities_index==entity[1]:
                        for val in reversed(range(grade)):
                            if entities[val]=='O':
                                entities[val] = f'B-{entity[0]}'
                                added = 1
                                break
                    else:
                        for val in reversed(range(grade)):
                            if entities[val]=='O':
                                entities[val] = f'I-{entity[0]}' 
                                added = 1
                                break 

                    if not added:
                        print(line)
                        print(entity)
                
            entities_index+=1
            string = '\t'.join(entities)
            if i==len(lines)-1 and k==len(annotations)-1: new_text.write(f"{token}\t{string}")
            else: new_text.write(f"{token}\t{string}\n")
        
        if k!=len(annotations)-1: new_text.write('\n')
    
    create_partitions('../neural_layered_files/entities.conll', '../neural_layered_files/train.conll', '../neural_layered_files/dev.conll', '../neural_layered_files/test.conll')




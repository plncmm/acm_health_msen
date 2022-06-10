import argparse 
import os 
import codecs 
import math
import sys
import logging
import glob
from tqdm import tqdm
from brat_to_conll import create_single_entity_data, create_partitions, convert_to_conll
from tokenizer import fix_tokens
from utils import boolean_string

def get_files(path):

  # Plain text files.
  text_files = glob.glob(f"{path}/*.txt")
  text_files.sort()

  # Annotation files.
  ann_files = glob.glob(f"{path}/*.ann")
  ann_files.sort()

  # Verifying that each text file has an associated annotation file.
  for t, a in zip(text_files, ann_files):
      assert(t.split('/')[-1].split('.')[0] == a.split('/')[-1].split('.')[0])
    
  return text_files, ann_files

def get_content(text_files, ann_files):
  
  texts, annotations = [], []

  for text_file, ann_file in zip(text_files, ann_files):
    # We open the file and save the content of the plain text.
    text_content = open(text_file, 'r').read()
    texts.append((text_file, text_content))

    # We open the file and save the content of the annotation.
    ann_content = open(ann_file, 'r').read() 
    annotations.append((ann_file, ann_content))
    
  return texts, annotations

if __name__ == "__main__":
    entity_types = ['Disease', 'Medication', 'Body_Part', 'Abbreviation', 'Finding', 'Procedure', 'Family_Member']
    if not os.path.exists('wl_files/'): os.mkdir('wl_files/') 
    
    for entity in entity_types:
        if not os.path.exists(f'wl_files/{entity}'): os.mkdir(f'wl_files/{entity}')

    actual_path = os.path.abspath(os.path.dirname(__file__))
    output_path = os.path.join(actual_path, f'wl_files/entities.conll')
    text_files, ann_files = get_files('../raw_data/')
    referrals, annotations = get_content(text_files, ann_files)

        

    convert_to_conll(referrals, annotations, entity_types, output_path)

    fix_tokens(output_path, f'wl_files/entities_fixed.conll')
    create_single_entity_data(f'wl_files/entities_fixed.conll')
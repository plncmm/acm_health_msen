import argparse 
import os 
import codecs 
import math
import sys
sys.path.append('..')
from brat_to_conll import create_single_entity_data, create_partitions, convert_to_conll
from request import samples_loader_from_minio
from tokenizer import fix_tokens
from utils import boolean_string
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type = str, required = True)
    parser.add_argument('--access_key', type = str, required = True)
    parser.add_argument('--secret_key', type = str, required = True)
    parser.add_argument('--n_annotations', type = int, default = 5000, required = False)
    parser.add_argument('--tokenizer', type = str, default = 'spacy', required = False) 
    parser.add_argument('--lower_tokens', default=False, type=boolean_string)
    parser.add_argument('--no_accent_marks', default=False, type=boolean_string)
    parser.add_argument('--include_path', default=False, type=boolean_string)
    parser.add_argument('--output_filename', type = str, default = 'wl_entities', required = False)
    parser.add_argument(
        '-t', 
        '--types', 
        default=None,
        metavar='TYPE', 
        nargs='*', 
        help='Filter entities to given types')
    args = parser.parse_args()  
    server = args.server
    access_key = args.access_key
    secret_key = args.secret_key
    n_annotations = args.n_annotations
    tokenizer = args.tokenizer
    lower_tokens = args.lower_tokens
    no_accent_marks = args.no_accent_marks
    include_path = args.include_path
    output_filename = args.output_filename
    entity_types = args.types
    if not os.path.exists('../wl_files'): os.mkdir('../wl_files') 
    for entity in entity_types:
        if not os.path.exists(f'../wl_files/{entity}'): os.mkdir(f'../wl_files/{entity}')

    actual_path = os.path.abspath(os.path.dirname(__file__))
    output_path = os.path.join(actual_path, f'../wl_files/{output_filename}_not_fixed.conll')
    referrals, annotations = samples_loader_from_minio(server, access_key, secret_key, n_annotations)
    logging.info('Creating files in conll format..')
    convert_to_conll(referrals, annotations, entity_types, tokenizer, lower_tokens, no_accent_marks, include_path, output_path)
    fix_tokens(output_path, f'../wl_files/{output_filename}.conll')
    create_partitions(f'../wl_files/{output_filename}.conll', f'../wl_files/{output_filename}_train.conll', f'../wl_files/{output_filename}_dev.conll', f'../wl_files/{output_filename}_test.conll')
    create_single_entity_data(f'../wl_files/{output_filename}.conll')
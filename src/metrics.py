from nestednereval.utils import read_iob2_prediction_file, merge_predictions
from nestednereval.metrics import nested_ner_metrics
import yaml

if __name__=='__main__':
    with open('../config.yaml') as file:
        config = yaml.safe_load(file)

    entity_types = ['Disease', 'Medication', 'Body_Part', 'Abbreviation', 'Finding', 'Procedure', 'Family_Member']
    chunks = []

    for entity in entity_types:
        entity_chunks = read_iob2_prediction_file(f"{config['output_path']}/{entity}/test.tsv") 
        chunks.append(entity_chunks)

    entities = merge_predictions(chunks) 

    print()
    nested_ner_metrics(entities) 
    print() 
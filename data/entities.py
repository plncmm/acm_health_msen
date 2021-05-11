from utils import simplify_entity

def get_nested_entities(annotation, referral, entity_types):  
    """ 
    Given a text and its annotation file, it returns all inner and outer entities annotated.
    """

    entities = []

    for line in annotation.splitlines():
        entity_info = {}
        entity = line.split()

        if entity[0].startswith('T') and not ';' in entity[3] and (entity[1] in entity_types or simplify_entity(entity[1]) in entity_types):
            entity_info['label'] = simplify_entity(entity[1]) if simplify_entity(entity[1]) in entity_types else entity[1]
            entity_info['start_idx'] = int(entity[2])
            entity_info['end_idx'] = int(entity[3])
            entity_info['text'] = referral[1][int(entity[2]): int(entity[3])] # Llego con la nueva entidad posiblemente a agregar.
            add = True # Booleano para saber si la incorporo o no 

            for entity_added in entities: # Por cada una de las entidades ya agregadas
                if entity_info['label']==entity_added['label'] and entity_info['start_idx']>=entity_added['start_idx'] and entity_info['end_idx']<=entity_added['end_idx']: # En caso que sea del mismo tipo y este anidada dentro de la otra, no se agrega.
                    add = False 
                    break

                elif entity_info['label']==entity_added['label'] and ((entity_info['start_idx']<entity_added['start_idx'] and entity_info['end_idx']>=entity_added['end_idx']) or (entity_info['start_idx']<=entity_added['start_idx'] and entity_info['end_idx']>entity_added['end_idx'])): 
                    add = False
                    entities.remove(entity_added)
                    entities.append(entity_info)
                    break
            if add and not colapse_with_others(entities, entity_info): 
                entities.append(entity_info)
    
    return entities

def get_flat_entities(annotations, referral):
    eliminate = {}
    for ann in annotations:
        for ann2 in annotations:
            if ann is ann2:
                continue
            if ann2['start_idx'] >= ann['end_idx'] or ann2['end_idx'] <= ann['start_idx']:
                continue 
            if eliminate.get((ann['label'], ann['start_idx'], ann['end_idx'])) or eliminate.get((ann['label'], ann2['start_idx'], ann2['end_idx'])):
                continue
            elim, keep = eliminate_and_keep(ann, ann2)
            eliminate[(elim['label'], elim['start_idx'], elim['end_idx'])] = True
    flat_entities = [anno for anno in annotations if not (anno['label'], anno['start_idx'], anno['end_idx']) in eliminate]
    return flat_entities

def colapse_with_others(entities, entity_info):
    for entity in entities:
        if (entity_info['start_idx']<entity['start_idx'] and entity_info['end_idx']>entity['start_idx'] and entity_info['end_idx']<entity['end_idx'])\
            or (entity_info['start_idx'] > entity['start_idx'] and entity_info['start_idx'] < entity['end_idx'] and entity_info['end_idx'] > entity['end_idx']):
            return True
    return False

def eliminate_and_keep(ann, ann2):
    if (ann['start_idx'], ann['end_idx']) == (ann2['start_idx'], ann2['end_idx']):
        return ann2, ann
    elif ann['end_idx']-ann['start_idx'] == ann2['end_idx']-ann2['start_idx']:
        return ann2, ann
    else:
        if ann['end_idx']-ann['start_idx'] < ann2['end_idx']-ann2['start_idx']:
            return ann, ann2
        else:
            return ann2, ann 
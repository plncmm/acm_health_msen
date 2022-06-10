def simplify_entity(entity):
    """
    Function used only for the Waiting List corpus, it generalizes entities so as not to have so many classes.
    
    Parameters:
    entity (string): Entity name.
    Returns:
    _ (string): Returns the simplified entity or the original depending on the entity type.
    """
    if entity in ["Laboratory_or_Test_Result", "Sign_or_Symptom", "Clinical_Finding"]:
        return "Finding"
    elif entity in ["Procedure", "Laboratory_Procedure", "Therapeutic_Procedure", "Diagnostic_Procedure"]:
        return "Procedure"
    return entity

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def windows_linux_ambiguity(referral, annotation, referral_name):
    for line in annotation.splitlines():
        entity = line.split()
        if entity[0].startswith('T') and not ';' in entity[3] and ' '.join(referral[int(entity[2]): int(entity[3])].split())!=' '.join(entity[4:]):
            return True
    return False


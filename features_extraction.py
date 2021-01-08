import pandas as pd
import itertools
from utils import WORK_FOR, read_annotations, read_lines, ENTITIES_TYPE
import sys
import spacy

from tqdm import tqdm

pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', None)


def extract_features(ent1, ent2, sent, sent_id):

    # Entity-based features
    embedding = ent1.root.vector
    embedding += ent2.root.vector

    # Word-based features
    embedding += nlp(sent.text[ent1.end_char:ent2.end_char-len(ent2.text)]).vector
    embedding += nlp(sent.text[:ent1.end_char-len(ent1.text)].split()[-1]).vector if len(sent.text[:ent1.end_char-len(ent1.text)]) > 0 else 0
    embedding += nlp(sent.text[ent2.end_char:].split()[0]).vector if len(sent.text[ent2.end_char:]) > 0 else 0

    # Syntactic features
    # features['constituent_path'] = None
    # features['basic_syntactic_chunk_path'] = None
    # features['typed_dependency_path'] = dependency_path(ent1, ent2)

    features = pd.Series(embedding, name=(sent_id, ent1.text, ent2.text))
    features['ent1_type'] = ent1.root.ent_type_
    features['ent2_type'] = ent2.root.ent_type_
    features['concat_type'] = features['ent1_type'] + features['ent2_type']
    features['y'] = 0
    return features


def dependency_path(ent1, ent2):
    dep_path = []

    tok = ent1.root
    while tok.dep_ != 'ROOT':
        dep_path.append((tok.text, tok.dep_))
        tok = tok.head

    i = len(dep_path)

    tok = ent2.root
    while True:
        dep_path.insert(i, (tok.text, tok.dep_))
        if tok.dep_ == 'ROOT':
            break
        tok = tok.head

    return dep_path


nlp = spacy.load('en_core_web_lg')
annotations = read_annotations(sys.argv[2])

c = 0
for sent_id, annots in annotations.items():
    for a in annots:
        if a[1] == WORK_FOR:
            # print(f'{sent_id} {ann}')
            c+=1
print(f'{WORK_FOR} annotations: {c}')

df = pd.DataFrame()
for sent_id, sent_str in tqdm(read_lines(sys.argv[1])):
    sent = nlp(sent_str)
    for ent1, ent2 in itertools.combinations(sent.ents, 2):
        if ent1.root.ent_type_ not in ENTITIES_TYPE:
            continue
        if ent2.root.ent_type_ not in ENTITIES_TYPE:
            continue
        if ent1.root.ent_type_ == ent2.root.ent_type_:
            continue
        features = extract_features(ent1, ent2, sent, sent_id)
        df = df.append(features)
df = pd.get_dummies(df, columns=['concat_type', 'ent1_type', 'ent2_type'])

print(f'dataframe size to pickle: {df.shape}')
df.to_pickle("df.pkl")
print(f'dataframe saved as: df.pkl')




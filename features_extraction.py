import pandas as pd
import itertools
from utils import read_lines, ENTITIES_TYPE, stop_words, WORK_FOR, read_annotations
import sys
import spacy
import numpy as np

from tqdm import tqdm

pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', None)


def extract_features(ent1, ent2, sent, sent_id):

    # Entity-based features
    ent_embedding = ent1.root.vector.copy()
    ent_embedding += ent2.root.vector.copy()

    # Word-based features
    words_embedding = sent[ent1.end:ent2.start].vector.copy() if len(sent[ent1.end:ent2.start]) > 0 else 0
    words_embedding += sent[ent1.start-1].vector.copy() if ent1.start-1 > 0 else 0
    words_embedding += sent[ent2.end].vector.copy() if ent2.end < len(sent) else 0

    # Syntactic features
    # features['constituent_path'] = None
    # features['basic_syntactic_chunk_path'] = None
    dep_embedding, dep_path = dependency_path(ent1, ent2)
    # print(dep_path_embedding)
    # sys.exit()

    embedding = np.concatenate((ent_embedding, words_embedding, dep_embedding))
    features = pd.Series(embedding, name=(sent_id, ent1.text, ent2.text))
    # features['ent1_type'] = ent1.root.ent_type_
    # features['ent2_type'] = ent2.root.ent_type_
    # features['concat_type'] = features['ent1_type'] + features['ent2_type']
    # features['dep_path'] = dep_path
    features['y'] = 0
    return features


def dependency_path(ent1, ent2):
    vec = np.zeros(300)
    dep_path = []

    tok = ent1.root
    while tok.dep_ != 'ROOT':
        dep_path.append((tok.dep_, 'IN'))
        vec += tok.vector.copy()
        tok = tok.head

    i = len(dep_path)

    tok = ent2.root
    while True:
        dep_path.insert(i, (tok.dep_, 'OUT'))
        vec += tok.vector.copy()
        if tok.dep_ == 'ROOT':
            break
        tok = tok.head

    return vec, str(dep_path).strip('[]')


nlp = spacy.load('en_core_web_lg')
annotations = read_annotations(sys.argv[2])

df = pd.DataFrame()
for sent_id, sent_str in tqdm(read_lines(sys.argv[1])):
    # if sent_id == 'sent667':
    #     print('aaa')
    sent = nlp(sent_str)
    for ent1, ent2 in itertools.combinations(sent.ents, 2):
        if ent1.root.ent_type_ not in ENTITIES_TYPE:
            continue
        if ent2.root.ent_type_ not in ENTITIES_TYPE:
            continue
        if ent1.root.ent_type_ == ent2.root.ent_type_:
            continue
        features = extract_features(ent1, ent2, sent, sent_id)

        for ann in annotations[sent_id]:
            if ann[1] == WORK_FOR:
                if ent1.root.text in ann[0] or ent1.root.text in ann[2]:
                    if ent2.root.text in ann[0] or ent2.root.text in ann[2]:
                        features['y'] = 1

        df = df.append(features)
# df = pd.get_dummies(df, columns=['concat_type', 'ent1_type', 'ent2_type', 'dep_path'])

print(f'response varibale (y) counts:\n{df["y"].value_counts()}')
print(f'dataframe size to pickle: {df.shape}')
df.to_pickle(sys.argv[3])
print(f'dataframe saved as: {sys.argv[3]}')




import pandas as pd
import itertools
from utils import read_lines, ENTITIES_TYPE, stop_words, WORK_FOR, read_annotations
import sys
import spacy
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer

from tqdm import tqdm

pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', None)


def extract_features(ent1, ent2, sent):
    embedding = ent1.root.vector.copy()
    embedding += ent2.root.vector.copy()
    for w in ent1:
        embedding += w.vector.copy()
    for w in ent2:
        embedding += w.vector.copy()
    for i in range(len(ent1) - 2 + 1):
        embedding += ent1[i:i + 2].vector.copy()
    for i in range(len(ent2) - 2 + 1):
        embedding += ent2[i:i + 2].vector.copy()
    embedding += sent[ent1.start-1].vector.copy() if ent1.start-1 > 0 else 0
    embedding += sent[ent1.end].vector.copy() if ent1.end < len(sent) else 0
    embedding += sent[ent2.start - 1].vector.copy() if ent2.start - 1 > 0 else 0
    embedding += sent[ent2.end].vector.copy() if ent2.end < len(sent) else 0
    for w in sent[ent1.end:ent2.start]:
        if not w.is_punct:
            embedding += w.vector.copy()

    features = {}
    # NER FEATURES
    features['ent1_type'] = ent1.root.ent_type_
    features['ent2_type'] = ent2.root.ent_type_
    features['ent1_ent2_type'] = ent1.root.ent_type_ + " " + ent2.root.ent_type_
    features['dep_path'] = dependency_path(ent1, ent2)

    return embedding, features


def dependency_path(ent1, ent2):
    ent1_path = []
    tok = ent1.root
    while tok.dep_ != 'ROOT':
        # ent1_path.append(tok.text)
        ent1_path.append('IN' + "_" + tok.dep_)
        tok = tok.head

    ent2_path = []
    tok = ent2.root
    while True:
        # ent2_path.insert(0, tok.text)
        if tok.dep_ == 'ROOT':
            break
        ent2_path.insert(0, 'OUT' + "_" + tok.dep_)
        tok = tok.head

    return str(ent1_path + ent2_path).strip('[]')


nlp = spacy.load('en_core_web_lg')
gold_annotations = read_annotations(sys.argv[2])
print(f'{WORK_FOR} input annotations: {sum([1 if a[1] == WORK_FOR else 0 for sent_id, annots in gold_annotations.items() for a in annots])}')

E = []
F = []
y = []
pred_annotations = []
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
        embedding, features = extract_features(ent1, ent2, sent)
        E.append(embedding)
        F.append(features)

        related = 0
        for ann in gold_annotations[sent_id]:
            if (ent1.root.text in ann[0] and ent2.root.text in ann[2]) or (ent2.root.text in ann[0] and ent1.root.text in ann[2]):
                related = 1
        y.append(related)

        pred_ann = f'{sent_id}\t'\
                   f'{ent1.text if ent1.root.ent_type_ == "PERSON" else ent2.text}\t'\
                   f'{WORK_FOR}\t'\
                   f'{ent1.text if ent1.root.ent_type_ == "ORG" else ent2.text}\t'\
                   f'( {sent.text} )'
        pred_annotations.append(pred_ann)

if len(sys.argv) == 5:
    with open(sys.argv[4], 'rb') as f:
        v = pickle.load(f)
    X = v.transform(F)
else:
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(F)
    with open('vectorizer', 'wb') as f:
        pickle.dump(v, f, pickle.HIGHEST_PROTOCOL)

df = pd.DataFrame(E)
df = pd.concat([df, pd.DataFrame(X, columns=v.feature_names_)], axis=1)
df.index = pred_annotations
y = np.array(y)

print(f'response varibale (y) counts:\n{np.unique(y, return_counts=True)[1]}')
print(f'dataframe size: {df.shape}')

with open(sys.argv[3], 'wb') as f:
    pickle.dump((df, y), f, pickle.HIGHEST_PROTOCOL)
print(f'(df, y) saved as: {sys.argv[3]}')



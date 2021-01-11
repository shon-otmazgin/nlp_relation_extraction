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

def get_before_after(ent, sent):
    before = '<START>'
    for i in range(1, len(sent)):
        if ent.start - i > 0 and not sent[ent.start - i].is_punct:
            before = sent[ent.start - i].text
            break

    after = '<END>'
    for i in range(len(sent)):
        if ent.end + i < len(sent) and not sent[ent.end + i].is_punct:
            after = sent[ent.end + i].text
            break
    return before, after

def words_between(ent1, ent2, sent):
    words = set([w.text for w in sent[ent1.end:ent2.start] if not w.is_punct]) if len(
        sent[ent1.end:ent2.start]) > 0 else set(['<EMPTY>'])
    return words

# def phrase_chunking(sent):
#     np_chunks = [np for np in sent.noun_chunks]
#     for w in sent:
#         print(w, ent='')
#         for np in np_chunks:
#             if w in np:
#                 print( 'NP', end='')
#                 break
#         print()

def extract_features(ent1, ent2, sent):
    features = {}

    # WORDS FEATURES
    # features['ent1_head'] = ent1.root.text
    # features['ent2_head'] = ent2.root.text
    # features['ent1_ent2_head'] = ent1.root.text + " " + ent2.root.text

    features['bow_ent1_ent2'] = set([w.text for w in ent1] + [w.text for w in ent2])

    features['before_ent1'], features['after_ent1'] = get_before_after(ent1, sent)
    features['before_ent2'], features['after_ent2'] = get_before_after(ent2, sent)

    features['words_between'] = words_between(ent1, ent2, sent)

    # features['ent1_pos'] = ent1.root.tag_
    # features['ent2_pos'] = ent2.root.tag_

    # NER FEATURES
    # features['ent1_type'] = ent1.root.ent_type_
    # features['ent2_type'] = ent2.root.ent_type_
    # features['ent1_ent2_type'] = ent1.root.ent_type_ + " " + ent2.root.ent_type_
    features['dep_path'] = dependency_path(ent1, ent2)

    # phrase_chunking(sent)

    return features


def dependency_path(ent1, ent2):
    ent1_path = []
    tok = ent1.root
    while tok.dep_ != 'ROOT':
        ent1_path.append(tok.text)
        ent1_path.append('IN' + "_" + tok.dep_)
        tok = tok.head

    ent2_path = []
    tok = ent2.root
    while True:
        ent2_path.insert(0, tok.text)
        if tok.dep_ == 'ROOT':
            break
        ent2_path.insert(0, 'OUT' + "_" + tok.dep_)
        tok = tok.head
    if not ent1_path:
        ent1_path.append(ent1.root.ent_type_)
    else:
        ent1_path[0] = ent1.root.ent_type_
    if not ent2_path:
        ent2_path.append(ent2.root.ent_type_)
    else:
        ent2_path[-1] = ent2.root.ent_type_
    return ent1_path + ent2_path


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
        features = extract_features(ent1, ent2, sent)
        E.append(np.concatenate((ent1.vector.copy(), ent2.vector.copy())))
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
    v = DictVectorizer(sparse=True)
    X = v.fit_transform(F)
    with open('vectorizer', 'wb') as f:
        pickle.dump(v, f, pickle.HIGHEST_PROTOCOL)

df = pd.concat([pd.DataFrame(E), pd.DataFrame(X.toarray(), columns=v.feature_names_)], axis=1)
df.index = pred_annotations
y = np.array(y)

print(f'response varibale (y) counts:\n{np.unique(y, return_counts=True)[1]}')
print(f'dataframe size: {df.shape}')

with open(sys.argv[3], 'wb') as f:
    pickle.dump((df, y), f, pickle.HIGHEST_PROTOCOL)
print(f'(df, y) saved as: {sys.argv[3]}')


